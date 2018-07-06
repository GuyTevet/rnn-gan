import tensorflow as tf
from config import FLAGS, BATCH_SIZE, LAMBDA
from model import get_generator, get_discriminator, params_with_name


def get_optimization_ops(disc_cost, gen_cost, global_step):
    gen_params = params_with_name('Generator')
    disc_params = params_with_name('Discriminator')
    print("Generator Params: %s" % gen_params)
    print("Disc Params: %s" % disc_params)
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                                                                             var_list=gen_params,
                                                                                             global_step=global_step)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                                                                              var_list=disc_params)
    return disc_train_op, gen_train_op


def get_substrings_from_gt(real_inputs, seq_length, charmap_len):
    train_pred = []
    for i in range(seq_length):
        train_pred.append(
            tf.concat([tf.zeros([BATCH_SIZE, seq_length - i - 1, charmap_len]), real_inputs[:, :i + 1]],
                      axis=1))

    all_sub_strings = tf.reshape(train_pred, [BATCH_SIZE * seq_length, seq_length, charmap_len])

    if FLAGS.LIMIT_BATCH:
        indices = tf.random_uniform([BATCH_SIZE], 1, all_sub_strings.get_shape()[0], dtype=tf.int32)
        all_sub_strings = tf.gather(all_sub_strings, indices)
        return all_sub_strings[:BATCH_SIZE]
    else:
        return all_sub_strings


def define_objective(charmap, real_inputs_discrete, seq_length):
    real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
    Generator = get_generator(FLAGS.GENERATOR_MODEL)
    Discriminator = get_discriminator(FLAGS.DISCRIMINATOR_MODEL)
    train_pred, inference_op = Generator(BATCH_SIZE, len(charmap), seq_len=seq_length, gt=real_inputs)

    real_inputs_substrings = get_substrings_from_gt(real_inputs, seq_length, len(charmap))

    disc_real = Discriminator(real_inputs_substrings, len(charmap), seq_length, reuse=False)
    disc_fake = Discriminator(train_pred, len(charmap), seq_length, reuse=True)
    disc_on_inference = Discriminator(inference_op, len(charmap), seq_length, reuse=True)

    disc_cost, gen_cost = loss_d_g(disc_fake, disc_real, train_pred, real_inputs_substrings, charmap, seq_length, Discriminator)
    return disc_cost, gen_cost, train_pred, disc_fake, disc_real, disc_on_inference, inference_op


def define_class_objective(charmap, real_inputs_discrete, real_class_discrete, seq_length, num_classes):
    real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
    Generator = get_generator(FLAGS.GENERATOR_MODEL)
    Discriminator = get_discriminator(FLAGS.DISCRIMINATOR_MODEL)
    train_pred, inference_op = Generator(BATCH_SIZE, len(charmap), seq_len=seq_length, num_classes=num_classes,
                                         gt=real_inputs, gt_class=real_class_discrete)

    real_inputs_substrings = get_substrings_from_gt(real_inputs, seq_length, len(charmap))

    disc_real, disc_real_class = Discriminator(real_inputs_substrings, len(charmap), seq_length, num_classes, reuse=False)
    disc_fake, disc_fake_class = Discriminator(train_pred, len(charmap), seq_length, num_classes, reuse=True)
    disc_on_inference, disc_on_inference_class = Discriminator(inference_op, len(charmap), seq_length, num_classes, reuse=True)

    disc_cost, gen_cost = loss_d_g_class(disc_fake, disc_fake_class, disc_real, disc_real_class, real_class_discrete, num_classes)

    return disc_cost, gen_cost, train_pred, disc_fake, disc_fake_class, \
           disc_real, disc_real_class, disc_on_inference, disc_on_inference_class, inference_op

def loss_d_g(disc_fake, disc_real, fake_inputs, real_inputs, charmap, seq_length, Discriminator):
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    gen_cost = -tf.reduce_mean(disc_fake)

    # WGAN lipschitz-penalty
    alpha = tf.random_uniform(
        shape=[tf.shape(real_inputs)[0], 1, 1],
        minval=0.,
        maxval=1.
    )
    differences = fake_inputs - real_inputs
    interpolates = real_inputs + (alpha * differences)
    gradients = tf.gradients(Discriminator(interpolates, len(charmap), seq_length, reuse=True), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    disc_cost += LAMBDA * gradient_penalty

    return disc_cost, gen_cost

def loss_d_g_class(disc_fake, disc_fake_class, disc_real, disc_real_class, gt_class, num_classes):

    # get loss for discriminator
    disc_cost_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))
    disc_cost_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))
    disc_cost = disc_cost_real + disc_cost_fake

    # get loss for generator
    gen_cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))

    # Calculate classifier cost
    gt_class_one_hot = tf.one_hot(gt_class, num_classes)

    classifier_real_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_real_class, labels=gt_class_one_hot))
    classifier_fake_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_fake_class, labels=gt_class_one_hot))
    classifier_cost = classifier_fake_cost + classifier_real_cost

    disc_cost += classifier_cost
    gen_cost += classifier_fake_cost

    return disc_cost, gen_cost