import sys

from tensorflow.python.training.saver import latest_checkpoint

from config import *
from language_helpers import generate_argmax_samples_and_gt_samples,generate_argmax_samples_and_gt_samples_class, inf_train_gen, decode_indices_to_string
from objective import get_optimization_ops, define_objective, define_class_objective
from summaries import define_summaries, \
    log_samples

sys.path.append(os.getcwd())

from model import *
import model_and_data_serialization
from runtime_process import Runtime_data_handler


# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
# fill in the path to the extracted files here!

def run(iterations, seq_length, is_first, charmap, inv_charmap, prev_seq_length):
    if len(DATA_DIR) == 0:
        raise Exception('Please specify path to data directory in single_length_train.py!')

    # lines, _, _ = model_and_data_serialization.load_dataset(seq_length=seq_length, b_charmap=False, b_inv_charmap=False,
    #                                                         n_examples=FLAGS.MAX_N_EXAMPLES)

    # instance data handler
    data_handler = Runtime_data_handler(h5_path=FLAGS.H5_PATH,
                                        json_path=FLAGS.H5_PATH.replace('.h5','.json'),
                                        seq_len=seq_length,
                                        # max_len=self.seq_len,
                                        # teacher_helping_mode='th_extended',
                                        use_var_len=False,
                                        batch_size=BATCH_SIZE,
                                        use_labels=False)

    #define placeholders
    real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, seq_length])
    real_classes_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    global_step = tf.Variable(0, trainable=False)

    # build graph according to arch
    if FLAGS.ARCH == 'default':
        disc_cost, gen_cost, fake_inputs, disc_fake, disc_real, disc_on_inference, inference_op = define_objective(charmap,
                                                                                                                real_inputs_discrete,
                                                                                                                seq_length)
        disc_fake_class = None
        disc_real_class = None
        disc_on_inference_class = None

        visualize_text = generate_argmax_samples_and_gt_samples
    elif FLAGS.ARCH == 'class_conditioned':
        disc_cost, gen_cost, fake_inputs, disc_fake, disc_fake_class, disc_real, disc_real_class,\
        disc_on_inference, disc_on_inference_class, inference_op = define_class_objective(charmap,
                                                                                            real_inputs_discrete,
                                                                                            real_classes_discrete,
                                                                                            seq_length,
                                                                                            num_classes=len(data_handler.class_dict))
        visualize_text = generate_argmax_samples_and_gt_samples_class

    merged, train_writer = define_summaries(disc_cost, gen_cost, seq_length)
    disc_train_op, gen_train_op = get_optimization_ops(disc_cost, gen_cost, global_step)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as session:

        session.run(tf.initialize_all_variables())
        if not is_first:
            print("Loading previous checkpoint...")
            internal_checkpoint_dir = model_and_data_serialization.get_internal_checkpoint_dir(prev_seq_length)
            model_and_data_serialization.optimistic_restore(session,
                                                            latest_checkpoint(internal_checkpoint_dir, "checkpoint"))

            restore_config.set_restore_dir(
                load_from_curr_session=True)  # global param, always load from curr session after finishing the first seq

        # gen = inf_train_gen(lines, charmap)
        data_handler.epoch_start(seq_len=seq_length)


        for iteration in range(iterations):
            start_time = time.time()

            # Train critic
            for i in range(CRITIC_ITERS):
                _data , _labels = data_handler.get_batch()
                if FLAGS.ARCH == 'class_conditioned':
                    _disc_cost, _, real_scores, real_class_scores = session.run(
                        [disc_cost, disc_train_op, disc_real, disc_real_class],
                        feed_dict={real_inputs_discrete: _data, real_classes_discrete: _labels})
                else:
                    _disc_cost, _, real_scores = session.run(
                        [disc_cost, disc_train_op, disc_real],
                        feed_dict={real_inputs_discrete: _data, real_classes_discrete: _labels})
                    real_class_scores = None


            # Train G
            for i in range(GEN_ITERS):
                # _data = next(gen)
                _data, _labels = data_handler.get_batch()
                _ = session.run(gen_train_op, feed_dict={real_inputs_discrete: _data, real_classes_discrete: _labels})

            print("iteration %s/%s"%(iteration, iterations))
            print("disc cost %f"%_disc_cost)

            # Summaries
            # if iteration % 1000 == 999:
            if iteration % 2 == 0:
                _data, _labels = data_handler.get_batch()
                summary_str = session.run(
                    merged,
                    feed_dict={real_inputs_discrete: _data, real_classes_discrete: _labels}
                )

                train_writer.add_summary(summary_str, global_step=iteration)
                fake_samples, samples_real_probabilites, fake_scores, fake_class_scores = visualize_text(session, inv_charmap,
                                                                                      fake_inputs,
                                                                                      disc_fake,
                                                                                      data_handler,
                                                                                      real_inputs_discrete,
                                                                                      real_classes_discrete,
                                                                                      feed_gt=True,
                                                                                      iteration=iteration,
                                                                                      seq_length=seq_length,
                                                                                      disc_class=disc_fake_class)

                log_samples(fake_samples, fake_scores, iteration, seq_length, "train", class_scores=fake_class_scores)
                log_samples(decode_indices_to_string(_data, inv_charmap), real_scores, iteration, seq_length, "gt", class_scores=real_class_scores)

                # inference
                test_samples, _, fake_scores, fake_class_scores = visualize_text(session, inv_charmap,
                                                              inference_op,
                                                              disc_on_inference,
                                                              data_handler,
                                                              real_inputs_discrete,
                                                              real_classes_discrete,
                                                              feed_gt=False,
                                                              iteration=iteration,
                                                              seq_length=seq_length,
                                                              disc_class=disc_on_inference_class)
                # disc_on_inference, inference_op
                if not FLAGS.ARCH == 'class_conditioned':
                    log_samples(test_samples, fake_scores, iteration, seq_length, "test")


            if iteration % FLAGS.SAVE_CHECKPOINTS_EVERY == FLAGS.SAVE_CHECKPOINTS_EVERY-1:
                saver.save(session, model_and_data_serialization.get_internal_checkpoint_dir(seq_length) + "/ckp")

        data_handler.epoch_end()

        saver.save(session, model_and_data_serialization.get_internal_checkpoint_dir(seq_length) + "/ckp")
        session.close()
