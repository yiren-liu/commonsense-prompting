import os

class Args():
    def __init__(self):
        self.block_size = 512
        self.do_train = True
        self.do_eval = False
        self.generation = False
        self.generate_and_eval = False
        self.evaluate_during_training = True
        self.per_gpu_train_batch_size = 40
        self.per_gpu_eval_batch_size = 80    
            
        # self.per_gpu_train_batch_size = 1
        # self.per_gpu_eval_batch_size = 1

        self.gradient_accumulation_steps = 1
        self.learning_rate = 2e-5  # RAW 2
        self.weight_decay = 0
        self.adam_epsilon = 1e-8  # RAW 8
        self.max_grad_norm = 1.0
        # self.num_train_epochs = 8  # raw 10
        self.num_train_epochs = 40  # raw 10
        self.max_steps = -1
        self.warmup_steps = 120  # raw 120
        self.logging_steps = 200
        self.save_steps = 200
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = False
        self.should_continue = False
        self.seed = 42  # raw 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = 'O1'

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

        self.d_model = 300
        # self.d_model = 600

        TAG = 'baseline'
        TAG += '__d_model__' + str(self.d_model)
        self.output_dir = os.path.join('checkpoints', 'FUDGE', TAG)
        # self.generation_dir = os.path.join('outputs', 'bart_generated', TAG)
        self.model_type = 'mymodel'
        # self.tokenizer_name = "facebook/bart-base"

        self.data_path = "./data/dataset"
        self.data_cache_dir = f'./cached/data/FUDGE/{TAG}/all_data'

        self.train_file_name = "trainWithStrategy_short.tsv"
        self.eval_file_name = "devWithStrategy_short.tsv"
        self.test_file_name = "testWithStrategy_short.tsv"
        self.train_comet_file = "trainComet.txt"
        self.eval_comet_file = "devComet.txt"
        self.test_comet_file = "testComet.txt"
        self.situation_train_comet_file = "trainComet_st.txt"
        self.situation_eval_comet_file = "devComet_st.txt"
        self.situation_test_comet_file = "testComet_st.txt"
        self.situation_train_file_name = "trainSituation.txt"
        self.situation_eval_file_name = "devSituation.txt"
        self.situation_test_file_name = "testSituation.txt"

        # self.strategy = False
        self.strategy = True
        # self.context = False
        self.context = True
        self.turn = False
        self.role = False

        self.target_model_path = './checkpoints/bart/baseline'
        self.model_cache_dir = f'./cached/models/FUDGE/{TAG}'



