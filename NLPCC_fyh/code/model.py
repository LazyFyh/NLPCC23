import math
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.modeling_utils import PreTrainedModel
import dgl
import dgl.nn.pytorch as dglnn

from collections import defaultdict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mymodel(BertPreTrainedModel):
    def __init__(self, config, gcn_layers=3, lambda_boundary=0, event_embedding_size=200):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        activation_func = nn.ReLU()
        self.transform_start = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_end = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_span = nn.Linear(3 * config.hidden_size, config.hidden_size)
        if event_embedding_size > 0:
            self.event_embedding = nn.Embedding(config.event_num, event_embedding_size)
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 2 + event_embedding_size, config.hidden_size),
                activation_func,
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, self.num_labels)
            )
        else:
            self.event_embedding = None
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                activation_func,
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, self.num_labels)
            )

        # GRAPH
        self.sent_embedding = nn.Parameter(torch.randn(config.hidden_size))
        self.rel_name_lists = ['sp-sp', 'snt-sp', 'snt-snt']
        self.gcn_layers = gcn_layers
        self.GCN_layers = nn.ModuleList([RelGraphConvLayer(config.hidden_size, config.hidden_size, self.rel_name_lists,
                                num_bases=len(self.rel_name_lists), activation=activation_func, self_loop=True, dropout=0.1)
                                for i in range(self.gcn_layers)])
        self.middle_layer = nn.Sequential(
            nn.Linear(config.hidden_size * (self.gcn_layers+1), config.hidden_size),
            activation_func,
            nn.Dropout(config.hidden_dropout_prob)
        )

        pos_loss_weight = getattr(config, 'pos_loss_weight', None)
        self.pos_loss_weight = torch.tensor([pos_loss_weight for _ in range(self.num_labels)])
        self.pos_loss_weight[0] = 1

        self.margin = 0.1
        self.temperature = 0.12
        self.lamda_cl = 0.05
        self.lamda_rl = 1

        self.init_weights()

    def select_rep(self, batch_rep, token_pos):
        B, L, dim = batch_rep.size()
        _, num = token_pos.size()
        shift = (torch.arange(B).unsqueeze(-1).expand(-1, num) * L).contiguous().view(-1).to(batch_rep.device)
        token_pos = token_pos.contiguous().view(-1)
        token_pos = token_pos + shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res.view(B, num, dim)

    def select_single_token_rep(self, batch_rep, token_pos):
        B, L, dim = batch_rep.size()
        shift = (torch.arange(B) * L).to(batch_rep.device)
        token_pos = token_pos + shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        event_ids=None,
        labels=None,
        spans=None,
        span_lens=None,
        label_masks=None,
        trigger_index=None,
        subwords_snt2spans=None,
        subwords_span2snts=None,
        trigger_snt_ids=None,
        belongingsnts=None,
        graphs=None,
        start_labels=None,
        end_labels=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        split=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)
        bsz, seq_len, hidsize = last_hidden_state.size()
        span_num = spans.size(1)

        snt_emb = None  
        for snt_num in range(5):
            token_idxs = belongingsnts.view(-1) == snt_num
            token_idxs = token_idxs.view(bsz, -1)
            snt_emb_bs = None
            for bs in range(bsz):
                single_snt_emb = None 
                for seq in range(seq_len):
                    if attention_mask[bs][seq] and token_idxs[bs][seq]:
                        if single_snt_emb is None:
                            single_snt_emb = last_hidden_state[bs][seq][:].clone().detach().view(1, hidsize) 
                        else:
                            single_snt_emb = torch.cat((single_snt_emb,
                                                        last_hidden_state[bs][seq][:].clone().detach().view(1, hidsize)),
                                                        dim=0)
                if single_snt_emb is None:
                    single_snt_emb = torch.zeros(hidsize).view(1, hidsize).to(device)
                single_snt_emb = torch.max(single_snt_emb, dim=0)[0].view(1, hidsize)

                if snt_emb_bs is None:
                    snt_emb_bs = single_snt_emb
                else:
                    snt_emb_bs = torch.cat((snt_emb_bs, single_snt_emb), dim=0)
            if snt_emb is None:
                snt_emb = snt_emb_bs.view(bsz, 1, hidsize)
            else:
                snt_emb = torch.cat((snt_emb, snt_emb_bs.view(bsz, 1, hidsize)), dim=1)

        # GRAPH
        graphs = []
        node_features = []
        for bs in range(bsz):
            d = defaultdict(list)

            # 1. sentence-sentence
            node_feature = snt_emb[bs].clone().detach()  
            node_feature += self.sent_embedding
            sent_num = node_feature.size(0)
            trigger_snt = trigger_snt_ids[bs].item()
            for i in range(sent_num):
                for j in range(sent_num):
                    if i != j and j == trigger_snt:
                        d[('node', 'snt-snt', 'node')].append((i, j))

            span_id = sent_num 
            if split == 'train':         
                # 2. sentence-span
                subwords_span2snt_bs = subwords_span2snts[bs]      
                for i, which_snt in enumerate(subwords_span2snt_bs): 
                    which_snt = which_snt.item()
                    span_b, span_e = spans[bs][i]
                    span_b = span_b.item()
                    span_e = span_e.item()
                    if span_b == 1 and span_e == 1:                    
                        break
                    span_emb = torch.max(last_hidden_state[bs][span_b:span_e+1][:].clone().detach(), dim=0)[0].view(1, hidsize)  
                    node_feature = torch.cat((node_feature, span_emb), dim=0)
                    d[('node', 'snt-sp', 'node')].append((which_snt, span_id))
                    d[('node', 'snt-sp', 'node')].append((span_id, which_snt))
                    span_id += 1

            # 3.span-span
            trigger_index_bs = trigger_index[bs]
            trigger_emb = last_hidden_state[bs][trigger_index_bs][:].clone().detach().view(1, hidsize)  # 1*768
            node_feature = torch.cat((node_feature, trigger_emb), dim=0)

            trigger_span_id = span_id
            d[('node', 'snt-sp', 'node')].append((trigger_snt, trigger_span_id))
            d[('node', 'snt-sp', 'node')].append((trigger_span_id, trigger_snt))
            for span_number in range(sent_num, trigger_span_id):
                d[('node', 'sp-sp', 'node')].append((span_number, trigger_span_id))
            for rel in self.rel_name_lists:
                if ('node', rel, 'node') not in d:
                    d[('node', rel, 'node')].append((0, 0))

            graph = dgl.heterograph(d)
            graphs.append(graph)
            node_features.append(node_feature)

        node_features_big = torch.cat(node_features, dim=0)
        graph_big = dgl.batch(graphs).to(device)
        feature_bank = [node_features_big]

        for GCN_layer in self.GCN_layers:
            node_features_big = GCN_layer(graph_big, {"node": node_features_big})["node"]
            feature_bank.append(node_features_big)
        feature_bank = torch.cat(feature_bank, dim=-1)
        node_features_big = self.middle_layer(feature_bank)    

        graphs = dgl.unbatch(graph_big)
        cur_idx = 0
        gcn_span_emb = []
        gcn_snt_emb = None 
        for bs, graph in enumerate(graphs):
            sent_num = snt_emb[bs].size(0)
            node_num = graphs[bs].number_of_nodes('node')
            if gcn_snt_emb is None:
                gcn_snt_emb = node_features_big[cur_idx:cur_idx + sent_num].view(1, sent_num, hidsize)
            else:
                gcn_snt_emb = torch.cat((gcn_snt_emb, node_features_big[cur_idx:cur_idx + sent_num].view(1, sent_num, hidsize)), dim=0)

            spans_emb = node_features_big[cur_idx + sent_num:cur_idx + node_num]
            gcn_span_emb.append(spans_emb)
            cur_idx += node_num

        loss = torch.zeros(1).to(device)
        trigger_snt_emb = None                         
        for bs in range(bsz):
            trigger_snt = trigger_snt_ids[bs].item()
            temp_snt_emb = gcn_snt_emb[bs][trigger_snt].clone().detach().view(1, 1, hidsize)
            if trigger_snt_emb is None:
                trigger_snt_emb = temp_snt_emb 
            else:
                trigger_snt_emb = torch.cat((trigger_snt_emb, temp_snt_emb), dim=0)
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        sim = cos(trigger_snt_emb, gcn_snt_emb)      

        margin = self.margin
        for bs in range(bsz):
            contain_span_snts = []
            uncontain_span_snts = []
            trigger_snt_id = trigger_snt_ids[bs].item()
            subwords_span2snt = subwords_span2snts[bs]
            for span2snt in subwords_span2snt:
                if span2snt.item() == 0:
                    break
                if span2snt.item() == trigger_snt_id:
                    continue
                contain_span_snts.append(span2snt.item())
        
            for i in range(5):
                if i != trigger_snt_id and i not in contain_span_snts:
                    uncontain_span_snts.append(i)
        
            for pos_idx in contain_span_snts:
                pos = sim[bs][pos_idx]
                for neg_idx in uncontain_span_snts:
                    neg = sim[bs][neg_idx]
                    if (neg - pos).item() > -margin:  
                        loss += self.lamda_rl * (neg - pos + margin)

        start_feature = self.transform_start(last_hidden_state) 
        end_feature = self.transform_end(last_hidden_state) 
        b_feature = self.select_rep(start_feature, spans[:,:,0])       
        e_feature = self.select_rep(end_feature, spans[:,:,1])              
        context = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).repeat(bsz, span_num, 1).to(device)   
        context_mask = (context>=spans[:,:,0:1]) & (context<=spans[:,:,1:])
        context_mask = context_mask.float()
        context_mask /= torch.sum(context_mask, dim=-1, keepdim=True)           
        context_feature = torch.bmm(context_mask, last_hidden_state)       

        span_feature = torch.cat((b_feature, e_feature, context_feature), dim=-1) 
        span_feature = self.transform_span(span_feature)       

        trigger_feature = None 
        for bs in range(bsz):
            span_embedding = gcn_span_emb[bs]
            temp_trigger_emb = span_embedding[-1][:].view(1, hidsize) 
            if trigger_feature is None:
                trigger_feature = temp_trigger_emb
            else:
                trigger_feature = torch.cat((trigger_feature, temp_trigger_emb), dim=-1)
        trigger_feature = trigger_feature.view(bsz, -1).unsqueeze(1).expand(-1, span_num, -1)  

        logits = torch.cat((
            span_feature, trigger_feature,
            self.event_embedding(event_ids).unsqueeze(1).expand(-1, span_num, -1)), dim=-1
        ) 

        logits = self.classifier(logits) 
        label_masks_expand = label_masks.unsqueeze(1).expand(-1, span_num, -1) 
        logits = logits.masked_fill(label_masks_expand == 0, -1e4) 
        if labels is not None: 
            loss_fct = CrossEntropyLoss(weight=self.pos_loss_weight.to(device))
            loss += loss_fct(logits.view(-1, self.num_labels), labels.contiguous().view(-1))

        temperature = self.temperature
        for bs in range(bsz):
            span_embedding = gcn_span_emb[bs]
            gold_span_num = span_embedding.shape[0] - 1
            trigger_gcn_emb = span_embedding[-1][:].view(1, hidsize)  
            gold_span_embedding = span_embedding[:-1][:]             
            neg_span_embedding = span_feature[bs][gold_span_num:][:] 

            trigger_gcn_emb = F.normalize(trigger_gcn_emb, dim=1)
            gold_span_embedding = F.normalize(gold_span_embedding, dim=1)
            neg_span_embedding = F.normalize(neg_span_embedding, dim=1)

            neg = torch.sum(torch.exp(F.cosine_similarity(trigger_gcn_emb, neg_span_embedding, dim=1) / temperature))
            for i in range(gold_span_num):
                pos_span_embedding = gold_span_embedding[i].view(1, hidsize)
                pos = torch.exp(F.cosine_similarity(trigger_gcn_emb, pos_span_embedding, dim=1) / temperature)

                loss += self.lamda_cl * -torch.log(pos/(neg+pos)) 

        if torch.isnan(loss):
            raise ValueError("nan!!")

        return {
            'loss': loss[0],
            'logits': logits,
            'spans': spans,
        }