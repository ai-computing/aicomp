#
# Copyright 2019 The Tensorflow Authors. All Rights Reserved
#
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.experimental import dtensor
from typing import Tuple



class Dense(tf.Module):

    def __init__(self, input_sz, output_sz, init_seed, weight_layout, activation=None):
        super().__init__()

        random_normal_initializer = tf.function(tf.random.stateless_normal)

        self.weight = dtensor.DVariable(dtensor.call_with_layout(random_normal_initializer, weight_layout, shape=[input_sz, output_sz], seed=init_seed))

        if activation is None:
            activation = lambda x:x

        self.activation = activation

        bias_layout = weight_layout.delete([0])

        self.bias = dtensor.DVariable(dtensor.call_with_layout(tf.zeros, bias_layout, [output_sz]))

    def __call__(self, x):
        y = tf.matmul(x, self.weight) + self.bias
        y = self.activation(y)

        return y


class MLP(tf.Module):

    def __init__(self, dense_layouts: Tuple[dtensor.Layout, dtensor.Layout]):
        super().__init__()

        self.dense1 = Dense(1200, 48, (1, 2), dense_layouts[0], activation=tf.nn.relu)
        self.dense2 = Dense(48, 2, (3, 4), dense_layouts[1])
        #self.dense1 = Dense(1200, 24, (1, 2), dense_layouts[0], activation=tf.nn.relu)
        #self.dense2 = Dense(24, 2, (3, 4), dense_layouts[1])

    def __call__(self, x):
        y = x
        y = self.dense1(y)
        y = self.dense2(y)
        return y



class Optimus_t:
    def __init__(self, dp_size, tp_size):
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.devices = self.configure_devices('GPU')

        self.mesh = dtensor.create_mesh([("batch", self.dp_size), ("model", self.tp_size)], devices=self.devices)
        self.model = MLP([dtensor.Layout([dtensor.UNSHARDED, "model"], self.mesh), dtensor.Layout(["model", dtensor.UNSHARDED], self.mesh)])

        self.learning_rate = tf.constant(1e-2)



    def configure_devices(self, device_type):
        phy_devices = tf.config.list_physical_devices(device_type)
        num_device = len(phy_devices)
        device_config = tf.config.LogicalDeviceConfiguration(memory_limit=40000)
        for n in range(num_device):
            tf.config.set_logical_device_configuration(phy_devices[n], [device_config])
        return [f'{device_type}:{i}' for i in range(num_device)]


    @tf.function
    def run(self, data, label):
        with tf.GradientTape() as tape:
            logits = self.model(data)
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label))
        parameters = self.model.trainable_variables
        gradients = tape.gradient(loss, parameters)
        for param, param_grad in zip(parameters, gradients):
            param.assign_sub(self.learning_rate * param_grad)

        loss_per_sample = loss / len(data)
        return loss_per_sample

    def reconfigure_tensor(self, x, layout):
        x = tf.convert_to_tensor(x)
        sharded_dims = []

        queue = [x]
        for axis, dim in enumerate(layout.sharding_specs):
            if dim == dtensor.UNSHARDED:
                continue
            num_splits = layout.shape[axis]
            queue = tf.nest.map_structure(lambda x: tf.split(x, num_splits, axis=axis), queue)
            sharded_dims.append(dim)

        components = []
        for locations in layout.mesh.local_device_locations():
            t = queue[0]
            for dim in sharded_dims:
                split_index = locations[dim]
                t = t[split_index]
            components.append(t)

        return dtensor.pack(components, layout)

    def get_batch(self, data, label):
        data = self.reconfigure_tensor(data, layout=dtensor.Layout(['batch', dtensor.UNSHARDED], self.mesh))
        label = self.reconfigure_tensor(label, layout=dtensor.Layout(['batch'], self.mesh))
        return data, label




optimus_t = Optimus_t(dp_size=4, tp_size=2)


dataset = tfds.load('imdb_reviews', split='train', shuffle_files=True, batch_size=64)
text_vectorization = tf.keras.layers.TextVectorization(output_mode='tf_idf', max_tokens=1200, output_sequence_length=None)
text_vectorization.adapt(data=dataset.map(lambda x: x['text']))

def vectorize(features):
    return text_vectorization(features['text']), features['label']

dataset_vectorized = dataset.map(vectorize)

num_epochs = 1

log_interval = 10

for epoch in range(num_epochs):
    i = 0
    for x,y in dataset_vectorized:
        data, label = optimus_t.get_batch(x, y)
        if i == 0:
            print(f"[{i}] data ==> {data}")
        loss = optimus_t.run(data, label)

        if i % log_interval == 0:
            print(f"Loss: {loss.numpy()}")
        i += 1


print(f" ----------------------------------------------------------")
print(f"model.dense1.weight: {optimus_t.model.dense1.weight}")
print(f" ----------------------------------------------------------")
print(f"model.dense2.weight: {optimus_t.model.dense2.weight}")
print(f" ----------------------------------------------------------")

