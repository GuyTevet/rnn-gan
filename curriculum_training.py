import model_and_data_serialization
from config import *
from single_length_train import run
from summaries import log_run_settings
from runtime_process import Runtime_data_handler

create_logs_dir()
log_run_settings()

# _, charmap, inv_charmap = model_and_data_serialization.load_dataset(seq_length=32, b_lines=False)

# instance data handler
data_handler = Runtime_data_handler(seq_len=1,
                                    h5_path=FLAGS.H5_PATH,
                                    json_path=FLAGS.H5_PATH.replace('.h5','.json'),
                                    use_labels=False)
charmap, inv_charmap = data_handler.tag_dict, data_handler.inv_tag



REAL_BATCH_SIZE = FLAGS.BATCH_SIZE

if FLAGS.SCHEDULE_SPEC == 'all' :
    stages = list(range(FLAGS.START_SEQ, FLAGS.END_SEQ))
else:
    split = FLAGS.SCHEDULE_SPEC.split(',')
    stages = list(map(int, split))

print('@@@@@@@@@@@ Stages : ' + ' '.join(map(str, stages)))

for i in range(len(stages)):
    prev_seq_length = stages[i-1] if i>0 else 0
    seq_length = stages[i]
    print(
    "**********************************Training on Seq Len = %d, BATCH SIZE: %d**********************************" % (
    seq_length, BATCH_SIZE))
    tf.reset_default_graph()
    if FLAGS.SCHEDULE_ITERATIONS:
        iterations = min((seq_length + 1) * FLAGS.SCHEDULE_MULT, FLAGS.ITERATIONS_PER_SEQ_LENGTH)
    else:
        iterations = FLAGS.ITERATIONS_PER_SEQ_LENGTH
    run(iterations, seq_length, seq_length == stages[0] and not (FLAGS.TRAIN_FROM_CKPT),
        charmap,
        inv_charmap,
        prev_seq_length)

    if FLAGS.DYNAMIC_BATCH:
        BATCH_SIZE = REAL_BATCH_SIZE / seq_length
