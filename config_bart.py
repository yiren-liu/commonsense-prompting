import os

class Args():
    def __init__(self):
        self.model_type = 'mymodel'
     #    self.model_name_or_path = './blender-small'
        self.model_name_or_path = "facebook/bart-large"
        self.config_name = "facebook/bart-large"
        self.tokenizer_name = "facebook/bart-large"
        # self.model_name_or_path = "facebook/bart-base"
        # self.config_name = "facebook/bart-base"
        # self.tokenizer_name = "facebook/bart-base"

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


        # self.model_cache_dir = './cached/models/bart/debug'
        self.model_cache_dir = './cached/models/bart/large'

        # self.data_cache_dir = './cached/data/bart/debug_0.2'
        self.block_size = 512
        self.do_train = True
        self.do_eval = False
        self.generation = False
        self.generate_and_eval = False
        self.evaluate_during_training = True
        self.per_gpu_train_batch_size = 20
        self.per_gpu_eval_batch_size = 40    
            
        # self.per_gpu_train_batch_size = 1
        # self.per_gpu_eval_batch_size = 1

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
        # self.no_cuda = True
        self.overwrite_output_dir = True
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
        # TAG = 'bartEncoderClassifier'
        # TAG = 'lm_with_FUDGE'

        # TAG = 'genStrategyWithClassifier'
        self.generate_strategy = True
        self.strategy_predictor = "lm"
        # self.strategy_predictor = "gts"
        # self.strategy_predictor = "classifier"
        self.classifier_alpha = 1.0
        # self.d_model = 768
        self.d_model = 1024
        # self.use_fudge = True
        self.use_fudge = False

        self.situation_train_file_name = "trainSituation.txt"
        self.situation_eval_file_name = "devSituation.txt"
        self.situation_test_file_name = "testSituation.txt"
        
        self.cometStep_train_file_name = "trainCometOnly_DialogueHistory_ind_lastStep.jsonl"
        self.cometStep_eval_file_name = "devCometOnly_DialogueHistory_ind_lastStep.jsonl"
        self.cometStep_test_file_name = "testCometOnly_DialogueHistory_ind_lastStep.jsonl"
        
        
        # TAG = 'lm_with_FUDGE_withAppendCOMET'
        # TAG = 'lm_with_FUDGE_noCOMET'
        # TAG = 'lm_with_FUDGE_noCOMET_oracle'
        # TAG = 'baseline_with_noFUDGE_noCOMET'
        # TAG = 'baseline_with_noFUDGE_withAppendCOMET'
        # TAG = 'baseline_with_FUDGE_withAppendCOMET'
        TAG = 'baseline_noFUDGE_withAppendCOMET_lm'

        self.overwrite_cache = True
        self.append_comet_to_input = True
        # self.append_comet_to_input = False
        # self.use_comet_template = False
        self.use_comet_template = True


        if self.append_comet_to_input:
            self.data_cache_dir = './cached/data/bart/add_context_add_strategy_add_cometStep'
        else:
            self.data_cache_dir = './cached/data/bart/add_context_add_strategy'
        
        
        self.do_train = True
        # self.do_train = False
        self.load_dir = os.path.join('checkpoints', 'bart', TAG)
        # self.load_dir = os.path.join('checkpoints', 'bart', 'baseline')
        self.fudge_model_path = os.path.join('checkpoints', 'FUDGE', 'baseline__d_model__300')

        if self.DEBUG:      
            TAG = 'debug'
            self.per_gpu_train_batch_size = 2
            self.per_gpu_eval_batch_size = 2
            self.data_path = "./data/dataset/sample_5"
            self.data_cache_dir = './cached/data/bart/debug_5'
            self.model_cache_dir = './cached/models/bart/debug'
            self.model_name_or_path = "facebook/bart-base"
            self.config_name = "facebook/bart-base"
            self.tokenizer_name = "facebook/bart-base"
            # self.load_dir = os.path.join('checkpoints', 'bart', 'debug')
            # self.load_dir = os.path.join('checkpoints', 'bart', 'baseline')
            self.load_dir = os.path.join('checkpoints', 'bart', 'bartEncoderClassifier')


        # TAG = 'baseline'
        # self.situation_train_file_name = "trainSituation.txt"
        # self.situation_eval_file_name = "devSituation.txt"
        # self.situation_test_file_name = "testSituation.txt"
        # self.data_cache_dir = './cached/data/bart/add_context_add_strategy'

        # TAG = 'relNoConstraint'
        # self.situation_train_file_name = "trainComet_st_relAll.txt"
        # self.situation_eval_file_name = "devComet_st_relAll.txt"
        # self.situation_test_file_name = "testComet_st_relAll.txt"
        # self.data_cache_dir = './cached/data/bart/add_contextCOMET'

        # TAG = 'relConstraint'
        # self.situation_train_file_name = "trainComet_st_relConstraint.txt"
        # self.situation_eval_file_name = "devComet_st_relConstraint.txt"
        # self.situation_test_file_name = "testComet_st_relConstraint.txt"
        # self.data_cache_dir = './cached/data/bart/add_contextCOMET_relConstraint'

        self.output_dir = os.path.join('checkpoints', 'bart', TAG)
        self.generation_dir = os.path.join('outputs', 'bart_generated', TAG)


    def __getitem__(self, name):
        return getattr(self, name)