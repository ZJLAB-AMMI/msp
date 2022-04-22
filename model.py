import torch
from torch import mean, stack, LongTensor, cat
from read_data import prepare_data, domain_slot_list, domain_slot_type_map, SampleDataset
from config import args, logger, DEVICE
from torch.nn import ReLU, Linear, Sequential, Module, ModuleDict
from transformers import RobertaModel, BertModel
from transformers import RobertaTokenizer, BertTokenizer


if 'roberta' in args['pretrained_model']:
    tokenizer = RobertaTokenizer.from_pretrained(args['pretrained_model'])
elif 'bert' in args['pretrained_model']:
    tokenizer = BertTokenizer.from_pretrained(args['pretrained_model'])
else:
    raise ValueError('')
no_value_assign_strategy = args['no_value_assign_strategy']
overwrite_cache = args['overwrite_cache']
lock_embedding_parameter = args['lock_embedding_parameter']
use_multi_gpu = args['multi_gpu']
mentioned_slot_pool_size = args['mentioned_slot_pool_size']


def unit_test():
    pretrained_model = args['pretrained_model']
    name = args['process_name']
    slot_value_index_dict, slot_index_value_dict, train_loader, dev_loader, test_loader = \
        prepare_data(overwrite=overwrite_cache)
    _ = HistorySelectionModel(name, pretrained_model, slot_value_index_dict)
    logger.info('feed success')


class HistorySelectionModel(Module):
    def __init__(self, name, pretrained_model, slot_value_index_dict, local_rank=None):
        super(HistorySelectionModel, self).__init__()
        self.name = name
        if use_multi_gpu:
            assert local_rank is not None
            self.target_id = local_rank
        else:
            self.target_id = DEVICE
        self.embedding_dim = args['encoder_d_model']
        self.encoder = PretrainedEncoder(pretrained_model)
        self.slot_value_index_dict = slot_value_index_dict
        # Gate dict, 4 for none, dont care, mentioned, hit
        # 暂且不分domain slot specific的参数
        self.gate_predict = ModuleDict()
        self.gate_attention_query = ModuleDict()
        self.gate_attention_key = ModuleDict()
        # self.gate_combine = ModuleDict()
        self.hit_parameter = ModuleDict()
        for domain_slot in domain_slot_type_map:
            self.gate_predict[domain_slot] = Linear(self.embedding_dim, 4)
            self.gate_attention_query[domain_slot] = \
                Sequential(Linear(self.embedding_dim, 16), ReLU(), Linear(16, 16), ReLU())
            self.gate_attention_key[domain_slot] = \
                Sequential(Linear(self.embedding_dim, 16), ReLU(), Linear(16, 16), ReLU())
            # self.gate_combine[domain_slot] = Linear(16, 16)
            if domain_slot_type_map[domain_slot] == 'classify':
                if no_value_assign_strategy == 'miss':
                    num_value = len(self.slot_value_index_dict[domain_slot])
                else:
                    num_value = len(self.slot_value_index_dict[domain_slot]) + 1
                self.hit_parameter[domain_slot] = Linear(self.embedding_dim, num_value)
            elif domain_slot_type_map[domain_slot] == 'span':
                self.hit_parameter[domain_slot] = Linear(self.embedding_dim, 2)
            else:
                raise ValueError('Error Value')
        # m for mentioned slot
        self.m_query_para_dict = ModuleDict()
        self.m_slot_para_dict = ModuleDict()
        self.m_combine_dict = ModuleDict()
        for domain_slot in domain_slot_list:
            self.m_query_para_dict[domain_slot] = \
                Sequential(Linear(self.embedding_dim, 16), ReLU(), Linear(16, 16), ReLU())
            self.m_slot_para_dict[domain_slot] = \
                Sequential(Linear(self.embedding_dim, 16), ReLU(), Linear(16, 16), ReLU())
            self.m_combine_dict[domain_slot] = Linear(16, 16)

        # 是否锁定embedding的值
        self.token_embedding = self.encoder.model.embeddings.word_embeddings
        if lock_embedding_parameter:
            self.token_embedding.weight.requires_grad = False

        # 由于turn, domain, slot, mentioned type需要经常用到，因此构建一下这些常用策略的embedding
        self.common_token_embedding_dict = None

    def get_common_token_embedding(self, slot_value_index_dict):
        common_token_list, common_token_embedding_dict = ['label', 'inform', 'dontcare'], {}
        for i in range(0, 30):
            common_token_list.append(str(i))
        for domain_slot in domain_slot_list:
            domain, slot = domain_slot.split('-')[0], domain_slot.split('-')[-1]
            common_token_list.append(domain)
            common_token_list.append(slot)
        for domain_slot in slot_value_index_dict:
            for value in slot_value_index_dict[domain_slot]:
                common_token_list.append(value)

        for key in common_token_list:
            token = LongTensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + key))).to(self.target_id)
            common_token_embedding_dict[key] = torch.FloatTensor(
                mean(self.token_embedding(token), dim=0, keepdim=True).detach().to('cpu').numpy()).to(self.target_id)
        token = LongTensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<pad>"))).to(self.target_id)
        common_token_embedding_dict["<pad>"] = torch.FloatTensor(
            mean(self.token_embedding(token), dim=0, keepdim=True).detach().to('cpu').numpy()).to(self.target_id)
        self.common_token_embedding_dict = common_token_embedding_dict

    def forward(self, data):
        """
        context token id shape [batch size, sequence length]
        """
        if self.common_token_embedding_dict is None:
            raise ValueError('')

        id_list, active_domain, active_slot, context_token = data[0], data[1], data[2], data[3]
        context_mask, mentioned_slot_list_dict = data[4].type(torch.uint8), data[9]
        mentioned_slot_list_mask_dict, str_mentioned_slot_list_dict = data[10], data[11]
        mentioned_slot_embedding_list_dict = self.get_mentioned_slots_embedding(
            mentioned_slot_list_dict, str_mentioned_slot_list_dict)

        encode = self.encoder(context_token, padding_mask=context_mask)
        predict_value_dict = {}
        # Choose the output of the first token ([CLS]) to predict gate and classification)
        # 预测假定hit情况下的预测值
        for domain_slot in domain_slot_list:
            slot_type, weight = domain_slot_type_map[domain_slot], self.hit_parameter[domain_slot]
            if slot_type == 'classify':
                predict_value_dict[domain_slot] = weight(encode[:, 1, :])
            else:
                predict_value_dict[domain_slot] = weight(encode)

        predict_mentioned_slot_dict = self.predict_mentioned_slot_value(
            encode[:, 2, :], mentioned_slot_embedding_list_dict, mentioned_slot_list_mask_dict)
        predict_gate_dict = self.predict_gate_value(encode[:, 0, :], mentioned_slot_embedding_list_dict,
                                                    mentioned_slot_list_mask_dict)

        return predict_gate_dict, predict_value_dict, predict_mentioned_slot_dict

    def predict_mentioned_slot_value(self, context, mentioned_slots_embedding_dict, mentioned_slot_list_mask_dict):
        predict_mentioned_slot_dict = {}
        for domain_slot in domain_slot_list:
            query = context
            query_weight = self.m_query_para_dict[domain_slot](query).unsqueeze(dim=2)
            key_weight = self.m_slot_para_dict[domain_slot](mentioned_slots_embedding_dict[domain_slot])
            key_weight = self.m_combine_dict[domain_slot](key_weight)
            predicted_value = torch.bmm(key_weight, query_weight).squeeze()
            predicted_value = (~mentioned_slot_list_mask_dict[domain_slot]) * -1e6 + predicted_value
            predict_mentioned_slot_dict[domain_slot] = predicted_value
        return predict_mentioned_slot_dict

    def predict_gate_value(self, context, mentioned_slots_embedding_dict, mentioned_slot_list_mask_dict):
        # 预测Gate值
        gate_predict_dict = {}
        for domain_slot in domain_slot_list:
            query = context
            mentioned_slots_embedding = mentioned_slots_embedding_dict[domain_slot]
            query_weight = self.gate_attention_query[domain_slot](query).unsqueeze(dim=2)
            key_weight = self.gate_attention_key[domain_slot](mentioned_slots_embedding)
            score = torch.bmm(key_weight, query_weight).squeeze()
            score = (~mentioned_slot_list_mask_dict[domain_slot]) * -1e6 + score
            weight = torch.softmax(score, dim=1).unsqueeze(dim=2)
            mentioned_slots_embedding = torch.sum(mentioned_slots_embedding * weight, dim=1, keepdim=True).squeeze()
            embedding = mentioned_slots_embedding + context
            # embedding = context
            gate_predict_dict[domain_slot] = self.gate_predict[domain_slot](embedding)
        return gate_predict_dict

    def get_mentioned_slots_embedding(self, mentioned_slot_list_dict, str_mentioned_slot_list_dict):
        # mentioned slots embedding获取，因为数据本身并不齐整(主要是部分value可能一次解析出多个token id)，因此只能这么做
        target_id = self.target_id
        mentioned_slots_embedding_dict = {}
        for domain_slot in domain_slot_list:
            mentioned_slots_embedding_dict[domain_slot] = []
            for sample_idx in range(len(mentioned_slot_list_dict[domain_slot])):
                sample_list = []
                str_mentioned_slot_list = str_mentioned_slot_list_dict[domain_slot][sample_idx]
                mentioned_slot_list = mentioned_slot_list_dict[domain_slot][sample_idx]
                for mentioned_slot, str_mentioned_slot in zip(mentioned_slot_list, str_mentioned_slot_list):
                    # 按照正常情况，这些token一定能找到命中的值
                    # turn = self.common_token_embedding_dict[str_mentioned_slot[0]]
                    # mentioned_type = self.common_token_embedding_dict[str_mentioned_slot[1]]
                    domain = self.common_token_embedding_dict[str_mentioned_slot[2]]
                    # slot = self.common_token_embedding_dict[str_mentioned_slot[3]]
                    if str_mentioned_slot[4] in self.common_token_embedding_dict:
                        value = self.common_token_embedding_dict[str_mentioned_slot[4]]
                    else:
                        # 注意，此处的domain_slot和domain slot其实可能分指两个不同的domain-slot pair。但是根据我们的设计
                        # 根据domain_slot做mapping并不会导致classify判定出问题
                        # 另一方面，尽管我们预先做了cache，但是在一种特定的情况下，classify case的数据结果也可能出现OOv
                        # 这种情况是mentioned slot value在act中，但是这个act inform并没有被采纳，这就会导致inform value
                        # 不会再classify value dict中被记下，从而导致OOV
                        # if domain_slot_type_map[domain_slot] == 'classify' and \
                        #         str_mentioned_slot[3] not in self.common_token_embedding_dict:
                        #     assert str_mentioned_slot[4] == 'inform'
                        value = mean(self.token_embedding(LongTensor(mentioned_slot[4]).to(target_id)), dim=0,
                                     keepdim=True)

                    # turn = mean(self.token_embedding(LongTensor(mentioned_slot[0]).to(target_id)),
                    # dim=0, keepdim=True)
                    # domain = mean(self.token_embedding(LongTensor(mentioned_slot[1]).to(target_id)), dim=0,
                    #               keepdim=True)
                    # slot = mean(self.token_embedding(LongTensor(mentioned_slot[2]).to(target_id)), dim=0,
                    # keepdim=True)
                    # value = mean(self.token_embedding(LongTensor(mentioned_slot[3]).to(target_id)), dim=0,
                    # keepdim=True)
                    # type_ = mean(self.token_embedding(LongTensor(mentioned_slot[4]).to(target_id)), dim=0,
                    # keepdim=True)
                    # sample_list.append(mean(cat((value, mean(cat((turn, mentioned_type, domain, slot), dim=0), dim=0,
                    #                                          keepdim=True)), dim=0), dim=0, keepdim=True))
                    # 决定不用turn idx
                    sample_list.append(mean(value, dim=0, keepdim=True))
                    # sample_list.append(mean(cat((value, mean(cat((domain, slot), dim=0), dim=0,
                    #                                          keepdim=True)), dim=0), dim=0, keepdim=True))
                assert len(sample_list) == mentioned_slot_pool_size
                mentioned_slots_embedding_dict[domain_slot].append(stack(sample_list))
            mentioned_slots_embedding_dict[domain_slot] = stack(mentioned_slots_embedding_dict[domain_slot]).squeeze(2)
        return mentioned_slots_embedding_dict


class PretrainedEncoder(Module):
    def __init__(self, pretrained_model_name):
        super(PretrainedEncoder, self).__init__()
        self._model_name = pretrained_model_name
        if 'roberta' in pretrained_model_name:
            self.model = RobertaModel.from_pretrained(pretrained_model_name)
        elif 'bert' in pretrained_model_name:
            self.model = BertModel.from_pretrained(pretrained_model_name)
        else:
            ValueError('Invalid Pretrained Model')

    def forward(self, context, padding_mask):
        """
        :param context: [sequence_length, batch_size]
        :param padding_mask: [sequence_length, batch_size]
        :return: output:  [sequence_length, batch_size, word embedding]
        """
        # required format: [batch_size, sequence_length]
        if 'roberta' in self._model_name:
            assert context.shape[1] <= 512
        if 'roberta' in self._model_name or 'bert' in self._model_name:
            output = self.model(context, attention_mask=padding_mask)['last_hidden_state']
            return output
        else:
            ValueError('Invalid Pretrained Model')


if __name__ == '__main__':
    unit_test()

