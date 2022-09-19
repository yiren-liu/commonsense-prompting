import os

class Args():
    def __init__(self):
        # TAG = 'all_data'
        # TAG = 'sample_0.2'
        TAG = 'sample_100'
        # TAG = 'all_loss'
        # TAG = 'emotion'
        # TAG = 'ablation_strategy'
        # TAG = 'ablation_situation'
        # TAG = 'ablation_post'
        # nowtime = '10251756'
        self.output_dir = os.path.join('checkpoints', 'bart', TAG)
        self.generation_dir = os.path.join('outputs', 'bart_generated', TAG)
        self.model_type = 'mymodel'
     #    self.model_name_or_path = './blender-small'
        self.model_name_or_path = "facebook/bart-large"
        self.config_name = "facebook/bart-large"
        self.tokenizer_name = "facebook/bart-large"

        # self.data_path = "./data/dataset"
        self.data_path = "./data/dataset/sample_100"
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

        self.model_cache_dir = './cached/models/bart'
        self.data_cache_dir = './cached/data/bart'
        self.block_size = 512
        self.do_train = True
        self.do_eval = False
        self.generation = False
        self.generate_and_eval = False
        self.evaluate_during_training = True
        # self.per_gpu_train_batch_size = 6
        # self.per_gpu_eval_batch_size = 12        
        self.per_gpu_train_batch_size = 2
        self.per_gpu_eval_batch_size = 4
        self.gradient_accumulation_steps = 1
        self.learning_rate = 2e-5  # RAW 2
        self.weight_decay = 0
        self.adam_epsilon = 1e-8  # RAW 8
        self.max_grad_norm = 1.0
        # self.num_train_epochs = 8  # raw 10
        self.num_train_epochs = 10  # raw 10
        self.max_steps = -1
        self.warmup_steps = 120  # raw 120
        self.logging_steps = 1000
        self.save_steps = 1000
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
        self.strategy = False
        self.turn = False
        self.role = False