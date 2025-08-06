use_cuda = True  # use gpu or cpu
thread_num_per_gpu = 1  # Keep the value fixed

host_ip_address = "0.0.0.0"  # ip address, WEB service
port_number = 7861  # port number, WEB service
use_tree_structure = False  # if True, the comment tree should be included in the source data
sum_batch_size = 32  # used in the extractive UCS system
sum_model_type = "tuning"  # used in the extractive UCS system
llm_input_limit = 1024  # When OOM, can reduce this value
atep_llm_input_limit = 1024  # When ATEP applied, should specify the token number of the input sequence
llm_based_background = False  # use llm-based background or not
use_comment_span = False  # if True, each comment will be split into smaller spans

# mode
running_mode = 1  # 0 for application, 1 for eval, 2 for pipeline eval
clustering_only = False  # if True, not summarize for each comment cluster

# DB configuration
use_db = False  # database or file mode
host = "10.2.56.213"  # ip address of the DB
username = "root"  # username of the DB
passwd = "123456"  # pwd of the DB user
port = 3306  # port used by the DB
db_name = "dsta_db"  # name of the DB

# other user configurations
comment_truncation = False

# Development Configurations
clustering_thr = None  # can be adjusted to the target task [0.54, 0.9, 0.82, 200]

# project purpose
use_stateless = False
use_llm_me = False

# Hyper parameter
# use_ac_tt = True  # me_type 0: ME detection, time consuming  1: 1st sentence  2: top-three sentences close to doc  3: tt only (some cases do not have tts)
# use_ac_1st = False  # First 3 sentences as reference
# use_ac_t3 = True  # Top 3 sentence closest to the doc vector
# use_event = False
# use_who = False
# use_where = False
# use_when = False
# use_trigger = False
# use_outcome = False

use_ac_tt = True  # me_type 0: ME detection, time consuming  1: 1st sentence  2: top-three sentences close to doc  3: tt only (some cases do not have tts)
use_ac_1st = True  # First 3 sentences as reference
use_ac_t3 = False  # Top 3 sentence closest to the doc vector
use_event = True
use_who = True
use_where = True
use_when = False
use_trigger = False
use_outcome = False

use_ac_me = False  # deprecated, this one is based on extractive ME extraction