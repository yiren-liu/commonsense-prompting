import os

class Args():
    def __init__(self):
        self.model_type = 'mymodel'
     #    self.model_name_or_path = './blender-small'
        self.model_name_or_path = "ref/comet-atomic-2020/models/comet_atomic2020_bart/comet-atomic_2020_BART"
        self.config_name = "ref/comet-atomic-2020/models/comet_atomic2020_bart/comet-atomic_2020_BART"
        self.tokenizer_name = "ref/comet-atomic-2020/models/comet_atomic2020_bart/comet-atomic_2020_BART"

        self.data_path = "./data/dataset"
        # self.data_path = "./data/dataset/sample_100"
        # self.data_path = "./data/dataset/sample_0.2"

        self.train_file_name = "trainWithStrategy_short.tsv"
        self.eval_file_name = "devWithStrategy_short.tsv"
        self.test_file_name = "testWithStrategy_short.tsv"
        self.train_comet_file = "trainComet.txt"
        self.eval_comet_file = "devComet.txt"
        self.test_comet_file = "testComet.txt"
        self.situation_train_comet_file = "trainComet_st.txt"
        self.situation_eval_comet_file = "devComet_st.txt"
        self.situation_test_comet_file = "testComet_st.txt"

        self.strategy2id = {
            "[Question]": 0,
            "[Reflection of feelings]": 1,
            "[Information]": 2,
            "[Restatement or Paraphrasing]": 3,
            "[Others]": 4,
            "[Self-disclosure]": 5,
            "[Affirmation and Reassurance]": 6,
            "[Providing Suggestions]": 7,
            "[None]": 8,
        }

        self.comet_template = {
            "[oEffect]": "As a result, Person Y",
            "[oReact]": "As a result, PersonY feels",
            "[oWant]": "As a result, PersonY wants",
            "[xAttr]": "PersonX is seen as",
            "[xEffect]": 'As a result, PersonX',
            "[xIntent]": 'because PersonX wanted',
            "[xNeed]": 'but before, PersonX needed',
            "[xReact]": 'As a result, PersonX feels',
            "[xReason]": 'because',
            "[xWant]": 'As a result, PersonX wants'
        }


        self.model_cache_dir = './cached/models/bart-comet2020/'
        # self.model_cache_dir = './cached/models/bart-comet2020/add_contextCOMET'
        # self.data_cache_dir = './cached/data/bart-comet2020/no_context_no_strategy'
        # self.data_cache_dir = './cached/data/bart-comet2020/add_context_add_strategy'
        # self.data_cache_dir = './cached/data/bart-comet2020/add_contextCOMET'

        self.block_size = 512
        self.do_train = True
        # self.do_train = False
        # self.do_eval = False
        # self.generation = False
        # self.generate_and_eval = False
        self.evaluate_during_training = True
        self.per_gpu_train_batch_size = 20
        self.per_gpu_eval_batch_size = 40        
        # self.per_gpu_train_batch_size = 2
        # self.per_gpu_eval_batch_size = 2
        self.gradient_accumulation_steps = 1
        self.learning_rate = 2e-5  # RAW 2
        self.weight_decay = 0
        self.adam_epsilon = 1e-8  # RAW 8
        self.max_grad_norm = 1.0
        # self.num_train_epochs = 8  # raw 10
        self.num_train_epochs = 2  # raw 10
        self.max_steps = -1
        self.warmup_steps = 120  # raw 120
        self.logging_steps = 100
        self.save_steps = 100
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        #    self.no_cuda = True
        self.overwrite_output_dir = True
        self.overwrite_cache = False
        self.should_continue = False
        self.seed = 42  # raw 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = 'O1'

        # self.strategy = False
        self.strategy = True
        # self.context = False
        self.context = True
        self.turn = False
        self.role = False




        # self.DEBUG = True
        self.DEBUG = False


        # TAG = 'genStrategyWithGold'
        self.generate_strategy = True
        # self.strategy_predictor = "lm"
        # self.strategy_predictor = "gts"

        # self.strategy_predictor = "bert_classifier"
        # self.pretrained_predictor_dir = os.path.join('checkpoints', 'predictor', 'BERT__d_model__300')

        self.strategy_predictor = "classifier"
        self.classifier_alpha = 1.0
        # self.d_model = 768
        self.d_model = 1024
        self.use_fudge = True
        # self.use_fudge = False

        self.situation_train_file_name = "trainSituation.txt"
        self.situation_eval_file_name = "devSituation.txt"
        self.situation_test_file_name = "testSituation.txt"
        # self.data_cache_dir = './cached/data/bart-comet2020/add_context_add_strategy'
        
        self.cometStep_train_file_name = "trainCometOnly_DialogueHistory_ind_lastStep.jsonl"
        self.cometStep_eval_file_name = "devCometOnly_DialogueHistory_ind_lastStep.jsonl"
        self.cometStep_test_file_name = "testCometOnly_DialogueHistory_ind_lastStep.jsonl"
        

        # TAG = 'genStrategyWithGold'
        # TAG = 'baseline_FUDGE_withAppendCOMETverbalized_lm_decodeOnly'
        # TAG = 'baseline_FUDGE_withAppendCOMETverbalized_bert_classifier_decodeOnly'
        # TAG = 'baseline_noFUDGE_withAppendCOMETverbalized_Linear_classifier'
        # TAG = 'baseline_FUDGE_withAppendCOMETverbalized_gts_decodeOnly'
        TAG = 'baseline_FUDGE_withAppendCOMETverbalized_Linear_classifier_decodeOnly'


        self.overwrite_cache = True
        self.append_comet_to_input = True
        # self.append_comet_to_input = False
        # self.use_comet_template = False
        self.use_comet_template = True


        if self.append_comet_to_input:
            self.data_cache_dir = './cached/data/bart-comet2020/add_context_add_strategy_add_cometStep'
        else:
            self.data_cache_dir = './cached/data/bart-comet2020/add_context_add_strategy'
        

        # self.do_train = True
        self.do_train = False
        # self.load_dir = os.path.join('checkpoints', 'bart-comet2020', TAG)
        # self.load_dir = os.path.join('checkpoints', 'bart', 'baseline')
        # self.load_dir = os.path.join('checkpoints', 'bart-comet2020', 'baseline_noFUDGE_withAppendCOMETverbalized_lm')
        self.load_dir = os.path.join('checkpoints', 'bart-comet2020', 'baseline_noFUDGE_withAppendCOMETverbalized_Linear_classifier')

        self.fudge_model_path = os.path.join('checkpoints', 'FUDGE', 'bartCOMET__d_model__300')


        if self.DEBUG:       
            self.per_gpu_train_batch_size = 1
            self.per_gpu_eval_batch_size = 1
            self.data_path = "./data/dataset/sample_100"
            self.data_cache_dir = './cached/data/bart-comet2020/debug_100'
            self.model_cache_dir = './cached/models/bart-comet2020/debug'
            self.model_name_or_path = "facebook/bart-base"
            self.config_name = "facebook/bart-base"
            self.tokenizer_name = "facebook/bart-base"


        self.output_dir = os.path.join('checkpoints', 'bart-comet2020', TAG)
        self.generation_dir = os.path.join('outputs', 'bart-comet2020_generated', TAG)


    def __getitem__(self, name):
        return getattr(self, name)