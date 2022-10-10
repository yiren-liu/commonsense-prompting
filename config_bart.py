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
        self.num_train_epochs = 4  # raw 10
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

        self.DEBUG = True

        if self.DEBUG:       
            self.per_gpu_train_batch_size = 1
            self.per_gpu_eval_batch_size = 1
            self.data_path = "./data/dataset/sample_100"
            self.data_cache_dir = './cached/data/bart/debug_100'
            self.model_cache_dir = './cached/models/bart/debug'
            self.model_name_or_path = "facebook/bart-base"
            self.config_name = "facebook/bart-base"
            self.tokenizer_name = "facebook/bart-base"

        TAG = 'genStrategyWithGold'
        self.generate_strategy = True
        self.use_gts_strategy = True
        self.situation_train_file_name = "trainSituation.txt"
        self.situation_eval_file_name = "devSituation.txt"
        self.situation_test_file_name = "testSituation.txt"
        self.data_cache_dir = './cached/data/bart/add_context_add_strategy'
        self.do_train = False
        self.load_dir = os.path.join('checkpoints', 'bart', 'debug')

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