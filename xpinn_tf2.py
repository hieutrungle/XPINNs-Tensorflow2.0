from typing import Callable
import logging
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io
import scipy
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
import sys
import copy
import os
import tensorflow_addons as tfa
import utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.keras.mixed_precision.set_global_policy('float64')


np.random.seed(1234)
print(f"{tf.config.list_physical_devices()}; "
      f"Tennsorflow version: {tf.__version__}\n")


class WarmUpLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        power: float = 1.0,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * \
                tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_outputs = num_outputs

    def build(self, input_shape):
        initializer = tf.keras.initializers.GlorotNormal()
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]),
                                             self.num_outputs],
                                      initializer=initializer,
                                      trainable=True,
                                      )
        self.bias = self.add_weight("bias",
                                    shape=[self.num_outputs],
                                    trainable=True,)
        self.regularizers = self.add_weight("regularizers",
                                            initializer=tf.keras.initializers.Constant(
                                                value=0.05),
                                            trainable=True,)

    def call(self, inputs):
        return 20*self.regularizers*(tf.add(tf.matmul(inputs, self.kernel), self.bias))

    def get_config(self):
        base_config = super().get_config()
        base_config['num_outputs'] = self.num_outputs
        return base_config


# Building xpinn model
class XPINN(tf.keras.models.Model):

    def __init__(self, layers1, layers2, layers3, name="XPINN", **kwargs):
        super().__init__(name=name, **kwargs)

        self.layers1 = layers1
        self.layers2 = layers2
        self.layers3 = layers3

        self.net_u1 = self.create_net_tanh(layers=layers1, name="net_u_tanh")
        self.net_u2 = self.create_net_sin(layers=layers2, name="net_u_sin")
        self.net_u3 = self.create_net_cos(layers=layers3, name="net_u_cos")

        self.net_u1.build(input_shape=(None, 2))
        self.net_u2.build(input_shape=(None, 2))
        self.net_u3.build(input_shape=(None, 2))

    def create_net_sin(self, layers, name=None):
        net_u = tf.keras.models.Sequential(name=name)
        for i, layer in enumerate(layers[:-1]):
            net_u.add(CustomDenseLayer(layer, name=f"dense_{i}"))
            net_u.add(tf.keras.layers.Lambda(
                lambda x: tf.sin(x), name=f"sin_{i}"))
        net_u.add(tf.keras.layers.Dense(
            layers[-1], kernel_initializer='glorot_normal', name=f"dense_final"))
        return net_u

    def create_net_cos(self, layers, name=None):
        net_u = tf.keras.models.Sequential(name=name)
        for i, layer in enumerate(layers[:-1]):
            net_u.add(CustomDenseLayer(layer, name=f"dense_{i}"))
            net_u.add(tf.keras.layers.Lambda(
                lambda x: tf.cos(x), name=f"cos_{i}"))
        net_u.add(tf.keras.layers.Dense(
            layers[-1], kernel_initializer='glorot_normal', name=f"dense_final"))
        return net_u

    def create_net_tanh(self, layers, name=None):
        net_u = tf.keras.models.Sequential(name=name)
        for i, layer in enumerate(layers[:-1]):
            net_u.add(CustomDenseLayer(layer, name=f"dense_{i}"))
            net_u.add(tf.keras.layers.Lambda(
                lambda x: tf.tanh(x), name=f"tanh_{i}"))
        net_u.add(tf.keras.layers.Dense(
            layers[-1], kernel_initializer='glorot_normal', name=f"dense_final"))
        return net_u

    def _get_second_derivative(self, u, x, y):
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        return u_xx, u_yy

    def call(self, inputs):
        boundary_x, boundary_y, boundary_u, f1_x, f1_y, f2_x, f2_y, f3_x, f3_y, i1_x, i1_y, i2_x, i2_y = inputs

        # Boundary conditions
        boundary_pred = self.net_u1(
            tf.concat([boundary_x, boundary_y], axis=1))

        # Sub-Net1
        u1 = self.net_u1(tf.concat([f1_x, f1_y], axis=1))
        u1_xx, u1_yy = self._get_second_derivative(u1, f1_x, f1_y)

        # Sub-Net2
        u2 = self.net_u2(tf.concat([f2_x, f2_y], axis=1))
        u2_xx, u2_yy = self._get_second_derivative(u2, f2_x, f2_y)

        # Sub-Net3
        u3 = self.net_u3(tf.concat([f3_x, f3_y], axis=1))
        u3_xx, u3_yy = self._get_second_derivative(u3, f3_x, f3_y)

        # Sub-Net1, Interface 1
        u1_i1 = self.net_u1(tf.concat([i1_x, i1_y], axis=1))
        u1_i1_xx, u1_i1_yy = self._get_second_derivative(u1_i1, i1_x, i1_y)

        # Sub-Net2, Interface 1
        u2_i1 = self.net_u2(tf.concat([i1_x, i1_y], axis=1))
        u2_i1_xx, u2_i1_yy = self._get_second_derivative(u2_i1, i1_x, i1_y)

        # Sub-Net1, Interface 2
        u1_i2 = self.net_u1(tf.concat([i2_x, i2_y], axis=1))
        u1_i2_xx, u1_i2_yy = self._get_second_derivative(u1_i2, i2_x, i2_y)

        # Sub-Net3, Interface 2
        u3_i2 = self.net_u3(tf.concat([i2_x, i2_y], axis=1))
        u3_i2_xx, u3_i2_yy = self._get_second_derivative(u3_i2, i2_x, i2_y)

        # Average value (Required for enforcing the average solution along the interface)
        uavg_i1 = (u1_i1 + u2_i1)/2
        uavg_i2 = (u1_i2 + u3_i2)/2

        # Residuals
        f1 = u1_xx + u1_yy - (tf.exp(f1_x) + tf.exp(f1_y))
        f2 = u2_xx + u2_yy - (tf.exp(f2_x) + tf.exp(f2_y))
        f3 = u3_xx + u3_yy - (tf.exp(f3_x) + tf.exp(f3_y))

        # Residual continuity conditions on the interfaces
        fi1 = (u1_i1_xx + u1_i1_yy - (tf.exp(i1_x) + tf.exp(i1_y))) - \
            (u2_i1_xx + u2_i1_yy - (tf.exp(i1_x) + tf.exp(i1_y)))
        fi2 = (u1_i2_xx + u1_i2_yy - (tf.exp(i2_x) + tf.exp(i2_y))) - \
            (u3_i2_xx + u3_i2_yy - (tf.exp(i2_x) + tf.exp(i2_y)))

        # loss
        loss_boundary = 20 * \
            tf.reduce_mean(tf.math.squared_difference(
                boundary_u, boundary_pred))

        loss1 = 1*tf.reduce_mean(tf.square(f1)) \
            + 1*tf.reduce_mean(tf.square(fi1)) \
            + 1*tf.reduce_mean(tf.square(fi2)) \
            + 20*tf.reduce_mean(tf.math.squared_difference(u1_i1, uavg_i1)) \
            + 20*tf.reduce_mean(tf.math.squared_difference(u1_i2, uavg_i2))

        loss2 = 1*tf.reduce_mean(tf.square(f2)) \
            + 1*tf.reduce_mean(tf.square(fi1)) \
            + 20*tf.reduce_mean(tf.math.squared_difference(u2_i1, uavg_i1))

        loss3 = 1*tf.reduce_mean(tf.square(f3)) \
            + 1*tf.reduce_mean(tf.square(fi2)) \
            + 20*tf.reduce_mean(tf.math.squared_difference(u3_i2, uavg_i2))

        return loss_boundary, loss1, loss2, loss3

    @tf.function
    def train_step(self, ipnuts):
        with tf.GradientTape(persistent=True) as tape:
            loss_boundary, loss1, loss2, loss3 = self(ipnuts)
            loss = loss_boundary + loss1 + loss2 + loss3

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        return loss_boundary, loss1, loss2, loss3

    def predict_step(self, inputs):
        all_f1_coors, all_f2_coors, all_f3_coors = inputs
        u_pred1 = self.net_u1(all_f1_coors)
        u_pred2 = self.net_u2(all_f2_coors)
        u_pred3 = self.net_u3(all_f3_coors)
        return u_pred1, u_pred2, u_pred3

    def compile(self, **kwargs):
        super().compile(
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            **kwargs,
        )


def train(model, inputs, training_config, X_star1, X_star2, X_star3, u_exact1, u_exact2, u_exact3):

    epochs = training_config["epochs"]
    start_epoch = training_config["start_epoch"]
    print_freq = training_config["print_freq"]
    end_epoch = start_epoch + epochs

    all_losses = []
    all_l2_errors = []

    # Train model
    for epoch in range(start_epoch, end_epoch):
        losses = model.train_step(inputs)
        loss_boundary = losses[0]
        loss1 = losses[1]
        loss2 = losses[2]
        loss3 = losses[3]
        all_losses.append(losses)

        # # Sub-Net1
        # u_pred1 = model.net_u1(X_star1)
        # Sub-Net2
        u_pred2 = model.net_u2(X_star2)
        # Sub-Net3
        u_pred3 = model.net_u3(X_star3)

        # l2_error1 = np.linalg.norm(
        #     u_exact1-u_pred1, 2)/np.linalg.norm(u_exact1, 2)
        l2_error2 = np.linalg.norm(
            u_exact2-u_pred2, 2)/np.linalg.norm(u_exact2, 2)
        l2_error3 = np.linalg.norm(
            u_exact3-u_pred3, 2)/np.linalg.norm(u_exact3, 2)

        all_l2_errors.append([l2_error2, l2_error3])

        if epoch % print_freq == 0 and epoch != start_epoch:

            print(
                f"Epoch {epoch}/{epochs}"
                f" - loss_boundary: {loss_boundary:.2e}"
                f" - loss1: {loss1:.2e}"
                f" - loss2: {loss2:.2e}"
                f" - loss3: {loss3:.2e}"
                f" - l2_error2: {l2_error2:.2e} - l2_error3: {l2_error3:.2e}"
            )
    return all_losses, all_l2_errors


def load_raw_data(filepath):
    # Load training data (boundary points), residual and interface points from .mat file
    # All points are generated in Matlab
    data = scipy.io.loadmat(filepath)

    # residual points
    f1_x = np.transpose(np.array(data['x_f1']))
    f1_y = np.transpose(np.array(data['y_f1']))
    f2_x = np.transpose(np.array(data['x_f2']))
    f2_y = np.transpose(np.array(data['y_f2']))
    f3_x = np.transpose(np.array(data['x_f3']))
    f3_y = np.transpose(np.array(data['y_f3']))
    residual_coors = {
        'f1_x': f1_x,
        'f1_y': f1_y,
        'f2_x': f2_x,
        'f2_y': f2_y,
        'f3_x': f3_x,
        'f3_y': f3_y,
    }

    # interface points
    i1_x = np.transpose(np.array(data['xi1']))
    i1_y = np.transpose(np.array(data['yi1']))
    i2_x = np.transpose(np.array(data['xi2']))
    i2_y = np.transpose(np.array(data['yi2']))
    interface_coors = {
        'i1_x': i1_x,
        'i1_y': i1_y,
        'i2_x': i2_x,
        'i2_y': i2_y,
    }

    # boundary points
    xb = np.transpose(np.array(data['xb']))
    yb = np.transpose(np.array(data['yb']))
    ub = np.transpose(np.array(data['ub']))
    boundary_coors = {
        'xb': xb,
        'yb': yb,
        'ub': ub,
    }

    # Extract training data
    u_exact = np.transpose(np.array(data['u_exact']))
    u_exact2 = np.transpose(np.array(data['u_exact2']))
    u_exact3 = np.transpose(np.array(data['u_exact3']))
    u_exacts = {
        'u_exact': u_exact,
        'u_exact2': u_exact2,
        'u_exact3': u_exact3,
    }

    # Points in the whole domain
    x_total = np.transpose(np.array(data['x_total']))
    y_total = np.transpose(np.array(data['y_total']))

    return residual_coors, interface_coors, boundary_coors, u_exacts, x_total, y_total


def process_data(residual_coors, interface_coors, boundary_coors, N_f1, N_f2, N_f3, N_ub, N_i1, N_i2):
    # Load training data (boundary points), residual and interface points from .mat file
    # All points are generated in Matlab

    # residual points
    f1_x = residual_coors['f1_x']
    f1_y = residual_coors['f1_y']
    f2_x = residual_coors['f2_x']
    f2_y = residual_coors['f2_y']
    f3_x = residual_coors['f3_x']
    f3_y = residual_coors['f3_y']

    # interface points
    i1_x = interface_coors['i1_x']
    i1_y = interface_coors['i1_y']
    i2_x = interface_coors['i2_x']
    i2_y = interface_coors['i2_y']

    # boundary points
    xb = boundary_coors['xb']
    yb = boundary_coors['yb']
    ub = boundary_coors['ub']

    # Stack x_f and y_f into one matrix
    f1_coors = np.hstack((f1_x, f1_y))
    f2_coors = np.hstack((f2_x, f2_y))
    f3_coors = np.hstack((f3_x, f3_y))

    # Stack xi and yi into one matrix
    f1_interface_coors = np.hstack((i1_x, i1_y))
    f2_interface_coors = np.hstack((i2_x, i2_y))

    # Stack xb and yb into one matrix
    b_coors = np.hstack((xb, yb))

    all_f1_coors = copy.deepcopy(f1_coors)
    all_f2_coors = copy.deepcopy(f2_coors)
    all_f3_coors = copy.deepcopy(f3_coors)
    all_f_coors = {
        'all_f1_coors': all_f1_coors,
        'all_f2_coors': all_f2_coors,
        'all_f3_coors': all_f3_coors
    }

    # Get a small number of points from each subdomain
    # Basically, we are going to use these points to train our NN
    # Randomly select the residual points from sub-domains
    idx1 = np.random.choice(f1_coors.shape[0], N_f1, replace=False)
    f1_coors = f1_coors[idx1, :]

    idx2 = np.random.choice(f2_coors.shape[0], N_f2, replace=False)
    f2_coors = f2_coors[idx2, :]

    idx3 = np.random.choice(f3_coors.shape[0], N_f3, replace=False)
    f3_coors = f3_coors[idx3, :]

    # Randomly select boundary points
    idx4 = np.random.choice(b_coors.shape[0], N_ub, replace=False)
    b_coors = b_coors[idx4, :]
    ub = ub[idx4, :]

    # Randomly select the interface points along two interfaces
    idxi1 = np.random.choice(f1_interface_coors.shape[0], N_i1, replace=False)
    f1_interface_coors = f1_interface_coors[idxi1, :]

    idxi2 = np.random.choice(f2_interface_coors.shape[0], N_i2, replace=False)
    f2_interface_coors = f2_interface_coors[idxi2, :]

    # utils.plot_data(f1_coors, f2_coors, f3_coors, f1_interface_coors,
    #                 f2_interface_coors, b_coors)

    # Model inputs
    # Boundary data
    boundary_x = tf.cast(tf.expand_dims(
        b_coors[:, 0], axis=1), dtype=tf.float64)
    boundary_y = tf.cast(tf.expand_dims(
        b_coors[:, 1], axis=1), dtype=tf.float64)
    boundary_u = tf.cast(ub, dtype=tf.float64)

    # Residual data
    f1_x = tf.cast(tf.expand_dims(
        f1_coors[:, 0], axis=1), dtype=tf.float64)
    f1_y = tf.cast(tf.expand_dims(
        f1_coors[:, 1], axis=1), dtype=tf.float64)
    f2_x = tf.cast(tf.expand_dims(
        f2_coors[:, 0], axis=1), dtype=tf.float64)
    f2_y = tf.cast(tf.expand_dims(
        f2_coors[:, 1], axis=1), dtype=tf.float64)
    f3_x = tf.cast(tf.expand_dims(
        f3_coors[:, 0], axis=1), dtype=tf.float64)
    f3_y = tf.cast(tf.expand_dims(
        f3_coors[:, 1], axis=1), dtype=tf.float64)

    # Interface data
    i1_x = tf.cast(tf.expand_dims(
        f1_interface_coors[:, 0], axis=1), dtype=tf.float64)
    i1_y = tf.cast(tf.expand_dims(
        f1_interface_coors[:, 1], axis=1), dtype=tf.float64)
    i2_x = tf.cast(tf.expand_dims(
        f2_interface_coors[:, 0], axis=1), dtype=tf.float64)
    i2_y = tf.cast(tf.expand_dims(
        f2_interface_coors[:, 1], axis=1), dtype=tf.float64)

    inputs = [boundary_x, boundary_y, boundary_u, f1_x, f1_y,
              f2_x, f2_y, f3_x, f3_y, i1_x, i1_y, i2_x, i2_y]

    return inputs, all_f_coors


def main():
    # Boundary points from subdomian 1
    N_ub = 200

    # Residual points in three subdomains
    N_f1 = 5000
    N_f2 = 1800
    N_f3 = 1200

    # Interface points along the two interfaces
    N_i1 = 100
    N_i2 = 100

    # NN architecture in each subdomain
    layers1 = [30, 30, 1]
    layers2 = [20, 20, 20, 20, 1]
    layers3 = [25, 25, 25, 1]

    # Load data
    data_path = './DATA/XPINN_2D_PoissonEqn.mat'
    residual_coors, interface_coors, boundary_coors, u_exacts, \
        x_total, y_total = load_raw_data(data_path)
    train_inputs, all_f_coors = process_data(
        residual_coors, interface_coors, boundary_coors, N_f1, N_f2, N_f3, N_ub, N_i1, N_i2)

    all_f1_coors = all_f_coors['all_f1_coors']
    all_f2_coors = all_f_coors['all_f2_coors']
    all_f3_coors = all_f_coors['all_f3_coors']

    u_exact = u_exacts['u_exact']
    u_exact2 = u_exacts['u_exact2']
    u_exact3 = u_exacts['u_exact3']

    # Model
    model = XPINN(layers1, layers2, layers3)

    # Training parameters
    training_config = {}
    training_config["epochs"] = 1_000
    training_config["print_freq"] = 20
    training_config["start_epoch"] = 1
    training_config["learning_rate"] = 1e-3
    training_config["weight_decay"] = 1e-7

    # Optimizer
    warmup_portion = 0.1
    warmup_steps = int(training_config["epochs"]*warmup_portion)
    decay_steps = int(training_config["epochs"]*(1-warmup_portion))
    decay_schedule_fn = tf.keras.optimizers.schedules.CosineDecay(
        training_config["learning_rate"], decay_steps, alpha=0.98)
    lr_schedule = WarmUpLRSchedule(
        initial_learning_rate=training_config["learning_rate"],
        decay_schedule_fn=decay_schedule_fn,
        warmup_steps=warmup_steps, power=1.0, name="warmUpCosineDecay"
    )
    optimizer = tfa.optimizers.AdamW(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.99,
        weight_decay=training_config["weight_decay"])

    # Compile
    model.compile(optimizer=optimizer)

    # Training
    all_losses, all_l2_errors = train(model, train_inputs, training_config,
                                      all_f1_coors, all_f2_coors, all_f3_coors,
                                      u_exact, u_exact2, u_exact3)

    # Solution prediction
    u_pred1, u_pred2, u_pred3 = model.predict_step(
        [all_f1_coors, all_f2_coors, all_f3_coors])

    # Concatenating the solution from subdomains
    u_pred = np.concatenate([u_pred1, u_pred2, u_pred3])  # shape: (22387, 1)
    error_u_total = np.linalg.norm(np.squeeze(
        u_exact)-u_pred.flatten(), 2)/np.linalg.norm(np.squeeze(u_exact), 2)
    print(f'\nError u_total: {error_u_total}\n')

    # Prepare figure folder
    fig_folder = os.path.join(os.getcwd(), 'xpinn_tf2_figures')
    utils.mkdir_if_not_exist(fig_folder)

    # plot training process
    utils.plot_losses(training_config, all_losses, fig_folder)
    utils.plot_l2_error(training_config, all_l2_errors, fig_folder)
    utils.plot_results(
        u_exact, u_pred, all_f_coors, boundary_coors, interface_coors, fig_folder)

    # plt.show()


if __name__ == "__main__":
    main()
