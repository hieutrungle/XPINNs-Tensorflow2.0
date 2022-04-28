import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
import numpy as np
import os


def plot_data(f1_coors, f2_coors, f3_coors, f1_interface_coors, f2_interface_coors, boundary_coors, fig_folder):
    # Plot the training data
    fig = plt.figure()
    title = 'training_data'
    ax = fig.add_subplot(111)
    ax.scatter(f1_coors[:, 0], f1_coors[:, 1],
               c='r', marker='o', label='$f_1$')
    ax.scatter(f2_coors[:, 0], f2_coors[:, 1],
               c='y', marker='o', label='$f_2$')
    ax.scatter(f3_coors[:, 0], f3_coors[:, 1],
               c='g', marker='o', label='$f_3$')
    ax.scatter(f1_interface_coors[:, 0], f1_interface_coors[:, 1],
               c='b', marker='o', label='$i_1$')
    ax.scatter(f2_interface_coors[:, 0], f2_interface_coors[:, 1],
               c='b', marker='o', label='$i_2$')
    ax.scatter(boundary_coors[:, 0], boundary_coors[:, 1],
               c='k', marker='x', label='$u_b$')
    ax.legend()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    fig.set_size_inches(w=9, h=7)
    plt.savefig(os.path.join(fig_folder, title+'.png'))


def plot_losses(training_config, all_losses, fig_folder):
    boundary_losses = [losses[0] for losses in all_losses]
    losses_1 = [losses[1] for losses in all_losses]
    losses_2 = [losses[2] for losses in all_losses]
    losses_3 = [losses[3] for losses in all_losses]

    fig = plt.figure()
    title = 'losses'
    plt.plot(range(1, training_config["epochs"]+1, 1), boundary_losses,
             'kx', linewidth=1, label='Boundary')
    plt.plot(range(1, training_config["epochs"]+1, 1), losses_1,
             'r-', linewidth=1, label='Sub-Net1')
    plt.plot(range(1, training_config["epochs"]+1, 1), losses_2,
             'b-.', linewidth=1, label='Sub-Net2')
    plt.plot(range(1, training_config["epochs"]+1, 1), losses_3,
             'g--', linewidth=1, label='Sub-Net3')
    plt.xlabel('$\#$ iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.title(title)
    fig.set_size_inches(w=9, h=7)
    plt.savefig(os.path.join(fig_folder, title+'.png'))


def plot_l2_error(training_config, all_l2_errors, fig_folder):
    l2_error2 = [errors[0] for errors in all_l2_errors]
    l2_error3 = [errors[1] for errors in all_l2_errors]

    fig = plt.figure()
    title = 'l2_error'
    plt.plot(range(1, training_config["epochs"]+1, 1), l2_error2,
             'r-', linewidth=1, label='Subdomain 2')
    plt.plot(range(1, training_config["epochs"]+1, 1), l2_error3,
             'b--', linewidth=1, label='Subdomain 3')
    plt.xlabel('$\#$ iterations')
    plt.ylabel('Rel. $L_2$ error')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.title(title)
    fig.set_size_inches(w=9, h=7)
    plt.savefig(os.path.join(fig_folder, title+'.png'))


def plot_results(u_exact, u_pred, all_f_coors, boundary_coors, interface_coors, fig_folder):

    def _partial_plot(title, values):
        i1_x = interface_coors['i1_x']
        i1_y = interface_coors['i1_y']
        i2_x = interface_coors['i2_x']
        i2_y = interface_coors['i2_y']

        # Stack xi and yi into one matrix
        f1_interface_coors = np.hstack((i1_x, i1_y))
        f2_interface_coors = np.hstack((i2_x, i2_y))

        fig = plt.figure()
        title = title
        ax = fig.subplots(1)
        tcf = ax.tricontourf(triang_total, values, 100, cmap='jet')
        ax.add_patch(Polygon(XX, closed=True, fill=True,
                     color='w', edgecolor='w'))
        tcbar = fig.colorbar(tcf)
        tcbar.ax.tick_params(labelsize=20)
        ax.set_xlabel('$x$', fontsize=24)
        ax.set_ylabel('$y$', fontsize=24)
        ax.set_title(title, fontsize=28)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        plt.plot(f1_interface_coors[:, 0:1], f1_interface_coors[:, 1:2],
                 'w-', markersize=2, label='Interface Pts')
        plt.plot(f2_interface_coors[:, 0:1], f2_interface_coors[:, 1:2],
                 'w-', markersize=2, label='Interface Pts')
        # fig.tight_layout()
        fig.set_size_inches(w=12, h=9)
        plt.savefig(os.path.join(fig_folder, title+'.png'), dpi=300)

    triang_total, XX = get_coordinates(all_f_coors, boundary_coors)
    # interface
    # interface points
    i1_x = interface_coors['i1_x']
    i1_y = interface_coors['i1_y']
    i2_x = interface_coors['i2_x']
    i2_y = interface_coors['i2_y']

    # Stack xi and yi into one matrix
    f1_interface_coors = np.hstack((i1_x, i1_y))
    f2_interface_coors = np.hstack((i2_x, i2_y))

    # Plot exact solution
    values = np.squeeze(u_exact)
    _partial_plot(title='exact_solution', values=values)

    # Plot predicted solution
    values = u_pred.flatten()
    _partial_plot(title='predicted_solution', values=values)

    # point-wise error
    values = abs(np.squeeze(u_exact)-u_pred.flatten())
    _partial_plot(title='point_wise_error', values=values)


def get_coordinates(all_f_coors, boundary_coors):
    all_f1_coors = all_f_coors['all_f1_coors']
    all_f2_coors = all_f_coors['all_f2_coors']
    all_f3_coors = all_f_coors['all_f3_coors']

    # Post-training Analysis
    xb = boundary_coors['xb']
    yb = boundary_coors['yb']
    # shape: (1, 2) last point of the boundary
    aa1 = np.array([[np.squeeze(xb[-1]), np.squeeze(yb[-1])]])
    aa2 = np.array([[1.8, np.squeeze(yb[-1])], [+1.8, -1.7], [-1.6, -1.7],
                   [-1.6, 1.55], [1.8, 1.55], [1.8, np.squeeze(yb[-1])]])
    aa3 = np.hstack((xb, yb))
    XX = np.vstack((aa3, aa2, aa1))

    # %% Needed for plotting
    f1_x, f1_y = all_f1_coors[:, 0:1], all_f1_coors[:, 1:2]
    f2_x, f2_y = all_f2_coors[:, 0:1], all_f2_coors[:, 1:2]
    f3_x, f3_y = all_f3_coors[:, 0:1], all_f3_coors[:, 1:2]
    x_tot = np.concatenate([f1_x, f2_x, f3_x])
    y_tot = np.concatenate([f1_y, f2_y, f3_y])
    triang_total = tri.Triangulation(x_tot.flatten(), y_tot.flatten())

    return triang_total, XX


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
