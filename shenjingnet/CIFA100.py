import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

conv_layers = [
    tf.keras.layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu'),  # channel 为64
    tf.keras.layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    tf.keras.layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'),  # channel 为64
    tf.keras.layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    tf.keras.layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'),  # channel 为64
    tf.keras.layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    tf.keras.layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),  # channel 为64
    tf.keras.layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    tf.keras.layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),  # channel 为64
    tf.keras.layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
]


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_val, y_val) = tf.keras.datasets.cifar100.load_data()
y = y.squeeze(axis=1)
y_val = y_val.squeeze(axis=1)
db = tf.data.Dataset.from_tensor_slices((x, y))
db = tf.data.Dataset.shuffle(db, buffer_size=10000).map(preprocess).batch(64)

db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
db_val = tf.data.Dataset.map(db_val, map_func=preprocess).batch(64)


def main():
    model = tf.keras.Sequential(conv_layers)

    x = tf.random.normal([4, 32, 32, 3])
    # out = model(x)
    # print(out.shape)
    fc_net = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(100, activation=None),
    ])

    model.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None, 512])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    variables = model.trainable_variables + fc_net.trainable_variables

    for epoch in range(50):

        for step, (x, y) in enumerate(db):

            with tf.GradientTape() as tape:
                out = model(x)
                out = tf.reshape(out, [-1, 512])

                logits = fc_net(out)

                y_one_hot = tf.one_hot(y, depth=100)
                loss = tf.losses.categorical_crossentropy(y_one_hot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 100 == 0:
                print(epoch, step, 'losses is', float(loss))


if __name__ == '__main__':
    main()




