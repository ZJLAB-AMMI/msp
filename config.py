import argparse
import torch
import os
import logging

# task and model setting
config_name = 'roberta'
model_type = 'base'
dataset = 'multiwoz21'
d_model = 768
lr = 0.00001
device = 'cuda:0'
mentioned_type = 'custom'
mode = 'train'  # train, eval
config = {
    'load_ckpt_path': '',
    'start_epoch': 0,  # = 0
    'process_name': 'history-mix-'+dataset+'-'+config_name+'-'+model_type + '-' + mentioned_type + '-mention',
    'process_no': '',
    'train_domain': 'hotel$train$restaurant$attraction$taxi',
    'test_domain': 'hotel$train$restaurant$attraction$taxi',
    'pretrained_model': config_name + '-' + model_type,
    'max_length': 512,
    'dataset': dataset,
    'batch_size': 4,
    'epoch': 20,
    'data_fraction': 0.1,
    'encoder_d_model': d_model,
    'learning_rate': lr,
    'device': device,
    'auxiliary_act_domain_assign': True,
    'delex_system_utterance': False,
    'use_multi_gpu': False,
    'no_value_assign_strategy': 'value',  # value
    'max_grad_norm': 1.0,
    'gate_weight': 0.6,
    'mentioned_weight': 0.2,
    'span_weight': 0.2,
    'classify_weight': 0.2,
    'overwrite_cache': False,
    'use_label_variant': True,
    'mode': mode,  # train, eval
    'lock_embedding_parameter': True,
    'mentioned_slot_pool_size': 3,
    'write_all_prediction': True
}


DEVICE = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='history_selection')
parser.add_argument('--load_ckpt_path', help='', default=config['load_ckpt_path'])
parser.add_argument('--use_label_variant', help='', default=config['use_label_variant'])
parser.add_argument('--write_all_prediction', help='', default=config['write_all_prediction'])
parser.add_argument('--device', help='', default=config['device'])
parser.add_argument('--mode', help='', default=config['mode'])
parser.add_argument('--dataset', help='', default=config['dataset'])
parser.add_argument('--lock_embedding_parameter', help='', default=config['lock_embedding_parameter'])
parser.add_argument('--start_epoch', help='', default=config['start_epoch'])
parser.add_argument('--process_name', help='', default=config['process_name'])
parser.add_argument('--overwrite_cache', help='', default=config['overwrite_cache'])
parser.add_argument('--mentioned_slot_pool_size', help='', default=config['mentioned_slot_pool_size'])
parser.add_argument('--train_domain', help='', default=config['train_domain'])
parser.add_argument('--gate_weight', help='', default=config['gate_weight'])
parser.add_argument('--span_weight', help='', default=config['span_weight'])
parser.add_argument('--classify_weight', help='', default=config['classify_weight'])
parser.add_argument('--mentioned_weight', help='', default=config['mentioned_weight'])
parser.add_argument('--delex_system_utterance', help='', default=config['delex_system_utterance'])
parser.add_argument('--multi_gpu', help='', default=config['use_multi_gpu'])
parser.add_argument('--data_fraction', help='', default=config['data_fraction'])
parser.add_argument('--epoch', help='', default=config['epoch'])
parser.add_argument('--learning_rate', help='', default=config['learning_rate'])
parser.add_argument('--encoder_d_model', help='', default=config['encoder_d_model'])
parser.add_argument('--batch_size', help='', default=config['batch_size'])
parser.add_argument('--pretrained_model', help='', default=config['pretrained_model'])
parser.add_argument('--local_rank', default=-1, type=int)  # 多卡时必须要有，由程序自行调用，我们其实不需要管
parser.add_argument('--test_domain', help='', default=config['test_domain'])
parser.add_argument('--max_grad_norm', help='', default=config['max_grad_norm'])
parser.add_argument('--no_value_assign_strategy', help='', default=config['no_value_assign_strategy'])
parser.add_argument('--max_len', help='', default=config['max_length'])
parser.add_argument('--auxiliary_act_domain_assign', help='', default=config['auxiliary_act_domain_assign'])
parser.add_argument("--weight_decay", default=0.0, type=float, help="")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="")
parser.add_argument("--warmup_proportion", default=0.1, type=float, help="")
args = vars(parser.parse_args())


DATA_TYPE_UTTERANCE, DATA_TYPE_SLOT, DATA_TYPE_BELIEF = 'utterance', 'slot', 'belief'
UNNORMALIZED_ACTION_SLOT = {'none', 'ref', 'choice', 'addr', 'post', 'ticket', 'fee', 'id', 'phone', 'car', 'open'}
UNK_token, PAD_token, SEP_token, CLS_token = '<unk>', '<pad>', '</s>', '<s>'


# resource path
multiwoz_dataset_folder = os.path.abspath('./'+dataset)
dialogue_data_path = os.path.join(multiwoz_dataset_folder, 'data.json')
dev_idx_path = os.path.join(multiwoz_dataset_folder, 'valListFile.json')
test_idx_path = os.path.join(multiwoz_dataset_folder, 'testListFile.json')
label_normalize_path = os.path.join(multiwoz_dataset_folder, 'label_map.json')
act_data_path = os.path.join(multiwoz_dataset_folder, 'dialogue_acts.json')
approximate_equal_path = os.path.join(multiwoz_dataset_folder, 'approximate_test.json')


model_checkpoint_folder = os.path.abspath('./model_check_point')
evaluation_folder = os.path.abspath('./evaluation')
cache_folder = os.path.abspath('./history_selection_cache')
if not os.path.exists(model_checkpoint_folder):
    os.makedirs(model_checkpoint_folder)
if not os.path.exists(evaluation_folder):
    os.makedirs(evaluation_folder)
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

cache_path = os.path.join(cache_folder, 'dialogue_data_cache_{}.pkl'.format(config['process_name']))

# dataset, time, epoch, general acc
medium_result_template = os.path.join(evaluation_folder, config['process_no'] +
                                      '_{}_{}_{}_{}.pkl')
ckpt_template = os.path.join(model_checkpoint_folder, config['process_no']+'_{}_{}.ckpt')
result_template = os.path.join(evaluation_folder, config['process_no'] + '_{}_{}_epoch_{}_{}.csv')


# logger
log_file_name = os.path.abspath('./log_{}_{}.txt'.format(config['process_no'], config['process_name']))
FORMAT = "%(asctime)s %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format=FORMAT, filename=log_file_name)
console_logger = logging.StreamHandler()
# console output format
stream_format = logging.Formatter("%(asctime)s %(process)d %(module)s %(lineno)d %(message)s")
# file output format
console_logger.setFormatter(stream_format)
logger.addHandler(console_logger)
logger.info("|------logger.info-----")

#
# MENTIONED_MAP_LIST = [
#     {'leaveat', 'arriveby', 'time'},
#     {'destination', 'departure', 'name'},
#     {'people'},
#     {'stay'},  # 指的是呆的时间
#     {'day'},  # 指具体星期几
#     {'food'},
#     {'pricerange'},
#     {'area'},
#     {'parking'},
#     {'stars'},
#     {'internet'},
#     {'type'}
# ]
#
POSSIBLE_MENTIONED_MAP_LIST_DICT = {
    'custom': {
        # source->target
        'taxi-leaveat': {"taxi-leaveat"},
        'taxi-destination': {'taxi-destination'},
        'taxi-departure': {},
        'taxi-arriveby': {'taxi-arriveby'},
        'restaurant-book-people': {'restaurant-book-people'},
        'restaurant-book-day': {'restaurant-book-day'},
        'restaurant-book-time': {'restaurant-book-time'},
        'restaurant-food': {},
        'restaurant-pricerange': {'restaurant-pricerange'},
        'restaurant-name': {},
        'restaurant-area': {},
        'hotel-book-people': {},
        'hotel-book-day': {},
        'hotel-book-stay': {},
        'hotel-name': {},
        'hotel-area': {},
        'hotel-stars': {},
        'hotel-parking': {},
        'hotel-pricerange': {},
        'hotel-type': {},
        'hotel-internet': {},
        'attraction-type': {},
        'attraction-name': {'attraction-name'},
        'attraction-area': {},
        'train-book-people': {},
        'train-arriveby': {},
        'train-destination': {},
        'train-departure': {},
        'train-leaveat': {},
        'train-day': {}
    },
    'self': {
        'taxi-leaveat': {"taxi-leaveat"},
        'taxi-destination': {'taxi-destination'},
        'taxi-departure': {'taxi-departure'},
        'taxi-arriveby': {'taxi-arriveby'},
        'restaurant-book-people': {'restaurant-book-people'},
        'restaurant-book-day': {'restaurant-book-day'},
        'restaurant-book-time': {'restaurant-book-time'},
        'restaurant-food': {'restaurant-food'},
        'restaurant-pricerange': {'restaurant-pricerange'},
        'restaurant-name': {'restaurant-name'},
        'restaurant-area': {'restaurant-area'},
        'hotel-book-people': {'hotel-book-people'},
        'hotel-book-day': {'hotel-book-day'},
        'hotel-book-stay': {'hotel-book-stay'},
        'hotel-name': {'hotel-name'},
        'hotel-area': {'hotel-area'},
        'hotel-stars': {'hotel-stars'},
        'hotel-parking': {'hotel-parking'},
        'hotel-pricerange': {'hotel-pricerange'},
        'hotel-type': {'hotel-type'},
        'hotel-internet': {'hotel-internet'},
        'attraction-type': {'attraction-type'},
        'attraction-name': {'attraction-name'},
        'attraction-area': {'attraction-area'},
        'train-book-people': {'train-book-people'},
        'train-arriveby': {'train-arriveby'},
        'train-destination': {'train-destination'},
        'train-departure': {'train-departure'},
        'train-leaveat': {'train-leaveat'},
        'train-day': {'train-day'}
    },
    'no': {
        'taxi-leaveat': {},
        'taxi-destination': {},
        'taxi-departure': {},
        'taxi-arriveby': {},
        'restaurant-book-people': {},
        'restaurant-book-day': {},
        'restaurant-book-time': {},
        'restaurant-food': {},
        'restaurant-pricerange': {},
        'restaurant-name': {},
        'restaurant-area': {},
        'hotel-book-people': {},
        'hotel-book-day': {},
        'hotel-book-stay': {},
        'hotel-name': {},
        'hotel-area': {},
        'hotel-stars': {},
        'hotel-parking': {},
        'hotel-pricerange': {},
        'hotel-type': {},
        'hotel-internet': {},
        'attraction-type': {},
        'attraction-name': {},
        'attraction-area': {},
        'train-book-people': {},
        'train-arriveby': {},
        'train-destination': {},
        'train-departure': {},
        'train-leaveat': {},
        'train-day': {}
    },
    'full': {
        'taxi-leaveat': {"taxi-leaveat"},
        'taxi-destination': {'taxi-destination', 'restaurant-name', 'attraction-name', 'hotel-name', 'train-departure'},
        'taxi-departure': {'taxi-departure'},
        'taxi-arriveby': {'taxi-arriveby'},
        'restaurant-book-people': {'restaurant-book-people'},
        'restaurant-book-day': {'restaurant-book-day'},
        'restaurant-book-time': {'restaurant-book-time', 'taxi-arriveby'},
        'restaurant-food': {'restaurant-food'},
        'restaurant-pricerange': {'restaurant-pricerange'},
        'restaurant-name': {'restaurant-name', 'taxi-destination', 'taxi-departure'},
        'restaurant-area': {'restaurant-area'},
        'hotel-book-people': {'hotel-book-people'},
        'hotel-book-day': {'hotel-book-day'},
        'hotel-book-stay': {'hotel-book-stay'},
        'hotel-name': {'hotel-name', 'taxi-destination', 'taxi-departure'},
        'hotel-area': {'hotel-area'},
        'hotel-stars': {'hotel-stars'},
        'hotel-parking': {'hotel-parking'},
        'hotel-pricerange': {'hotel-pricerange'},
        'hotel-type': {'hotel-type'},
        'hotel-internet': {'hotel-internet'},
        'attraction-type': {'attraction-type'},
        'attraction-name': {'attraction-name', 'taxi-destination', 'taxi-departure'},
        'attraction-area': {'attraction-area'},
        'train-book-people': {'train-book-people'},
        'train-arriveby': {'train-arriveby', 'taxi-leaveat'},
        'train-destination': {'train-destination', 'taxi-departure'},
        'train-departure': {'train-departure', 'taxi-destination'},
        'train-leaveat': {'train-leaveat', 'taxi-arriveby'},
        'train-day': {'train-day'}
    }
}

MENTIONED_MAP_LIST_DICT = POSSIBLE_MENTIONED_MAP_LIST_DICT[mentioned_type]


logger.info(MENTIONED_MAP_LIST_DICT)

# 以下内容均为直接复制
act_type = {
    'inform',
    'request',
    'select',  # for restaurant, hotel, attraction
    'recommend',  # for restaurant, hotel, attraction
    'not found',  # for restaurant, hotel, attraction
    'request booking info',  # for restaurant, hotel, attraction
    'offer booking',  # for restaurant, hotel, attraction, train
    'inform booked',  # for restaurant, hotel, attraction, train
    'decline booking'  # for restaurant, hotel, attraction, train
    # did not use four meaningless act, 'welcome', 'greet', 'bye', 'reqmore'
}
DOMAIN_IDX_DICT = {'restaurant': 0, 'hotel': 1, 'attraction': 2, 'taxi': 3, 'train': 4}
IDX_DOMAIN_DICT = {0: 'restaurant', 1: 'hotel', 2: 'attraction', 3: 'taxi', 4: 'train'}

SLOT_IDX_DICT = {'leaveat': 0, 'destination': 1, 'departure': 2, 'arriveby': 3, 'people': 4, 'day': 5, 'time': 6,
                 'food': 7, 'pricerange': 8, 'name': 9, 'area': 10, 'stay': 11, 'parking': 12, 'stars': 13,
                 'internet': 14, 'type': 15}

IDX_SLOT_DICT = {0: 'leaveat', 1: 'destination', 2: 'departure', 3: 'arriveby', 4: 'people', 5: 'day', 6: 'time',
                 7: 'food', 8: 'pricerange', 9: 'name', 10: 'area', 11: 'stay', 12: 'parking', 13: 'stars',
                 14: 'internet', 15: 'type'}


# Required for mapping slot names in dialogue_acts.json file
# to proper designations.
ACT_SLOT_NAME_MAP_DICT = {'depart': 'departure', 'dest': 'destination', 'leave': 'leaveat', 'arrive': 'arriveby',
                          'price': 'pricerange'}

ACT_MAP_DICT = {
    'taxi-depart': 'taxi-departure',
    'taxi-dest': 'taxi-destination',
    'taxi-leave': 'taxi-leaveat',
    'taxi-arrive': 'taxi-arriveby',
    'train-depart': 'train-departure',
    'train-dest': 'train-destination',
    'train-leave': 'train-leaveat',
    'train-arrive': 'train-arriveby',
    'train-people': 'train-book_people',
    'restaurant-price': 'restaurant-pricerange',
    'restaurant-people': 'restaurant-book-people',
    'restaurant-day': 'restaurant-book-day',
    'restaurant-time': 'restaurant-book-time',
    'hotel-price': 'hotel-pricerange',
    'hotel-people': 'hotel-book-people',
    'hotel-day': 'hotel-book-day',
    'hotel-stay': 'hotel-book-stay',
    'booking-people': 'booking-book-people',
    'booking-day': 'booking-book-day',
    'booking-stay': 'booking-book-stay',
    'booking-time': 'booking-book-time',
}
