import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
import numpy as np
import pickle

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

import sys
import time


def tf_train(data, savepath="./",  tolerance=1e-4, max_patience=15,
             batch_size=2500, hidden_nodes=100, max_epochs=500, validation=None, mode="cluster"):

    if savepath[-1] != "/":
        print("ERROR ON THE SAVEPATH")
        print("Appending / at the end.")
        savepath = savepath+"/"

        if os.path.exists(savepath) == True:
            print("Safe to save")
        else:
            print("ERROR SAVEPATH DOES NOT EXIST")
            print("RETURNING")
            return

    print("Model will be found at", savepath)

    x_train, y_train, x_test, y_test = data

    train_set = [np.split(x_train, batch_size), np.split(y_train, batch_size)]  # split data into batches
    best_loss = sys.maxsize

    number_of_patience = 0
    accuracy_train = []
    accuracy_test = []
    loss_list = []
    accuracy_validation = []
    features_num = len(x_train[0])
    print("Training Round ")

    # initialize graph before construct it otherwise saver will save empty graph in a loop
    graph = tf.Graph()
    with graph.as_default():

        x = tf.placeholder(tf.float32, [None, features_num], name='x')
        y = tf.placeholder(tf.float32, [None, 2], name='y')

        # first layer

        weights_l1 = tf.Variable(tf.truncated_normal([features_num, hidden_nodes], stddev=0.1), name='weight_l1')

        b_l1 = tf.Variable(tf.zeros([hidden_nodes]) + 0.01, name='biases_l1')
        l1 = tf.matmul(x, weights_l1) + b_l1
        l1 = tf.nn.relu(l1)

        # output layer
        weights_l2 = tf.Variable(tf.truncated_normal([hidden_nodes, 2], stddev=0.1), name='weight_l2')
        b_l2 = tf.Variable(tf.zeros([2]) + 0.01, name='biases_l2')
        output = tf.matmul(l1, weights_l2) + b_l2
        prediction = tf.nn.softmax(output, name='prediction')

        loss = tf.losses.log_loss(labels=y, predictions=prediction)
        train = tf.train.AdamOptimizer(0.001).minimize(loss)
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        saver = tf.train.Saver()
    begin = time.time()
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(max_epochs):
            for batch in range(len(train_set[0])):
                x_batch = train_set[0][batch]
                y_batch = train_set[1][batch]
                sess.run(train, feed_dict={x: x_batch, y: y_batch})
            acc_test = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
            acc_train = sess.run(accuracy, feed_dict={x: x_train, y: y_train})
            accuracy_test.append(acc_test)
            accuracy_train.append(acc_train)
            print("epoch ", epoch, "test accuracy is", acc_test)
            print("epoch ", epoch, "train accuracy is", acc_train)
            if validation != None:
                acc_validation = sess.run(accuracy, feed_dict={x: validation[0], y: validation[1]})
                accuracy_validation.append(acc_validation)
                print("epoch ", epoch, "validation accuracy is", acc_validation)

            current_loss = sess.run(loss, feed_dict={x: x_train, y: y_train})
            print("epoch ", epoch, " loss ", current_loss)
            loss_list.append(current_loss)
            # early stop procedure
            if current_loss > best_loss - tolerance:
                number_of_patience = number_of_patience + 1
            else:
                number_of_patience = 0
            if current_loss < best_loss:
                best_loss = current_loss
                model_name = "model_best_accuracy.ckpt"
                if mode == "cluster":
                    model_directory = str(savepath) + "best_model/"
                    os.makedirs(model_directory, exist_ok=True)
                    saver.save(sess, model_directory + model_name)
                    np.save(model_directory + 'acc_train.npy', accuracy_train)
                    np.save(model_directory + 'acc_test.npy', accuracy_test)
                    np.save(model_directory + 'loss.npy', loss_list)
                    if validation != None:
                        np.save(model_directory + 'acc_validation.npy', accuracy_validation)
                else:
                    saver.save(sess, savepath + model_name)


            if number_of_patience >= max_patience:
                print("Loss has not decrease significantly within last 10 iteration, stop at epoch", epoch)
                break

    print("Cost ", (time.time() - begin) / 60, "min")

    if mode != "cluster":
        if validation != None:
            return accuracy_train, accuracy_test, accuracy_validation, loss_list
        else:
            return accuracy_train, accuracy_test, loss_list
    else:
        return



if __name__ == "__main__":
    print("to implement")
    # args = sys.argv
    # if args[1] == "Calpha_iters":
    #     system = args[2]
    #     task_name = args[3]
    #     subset = int(args[4])
    #     subsets = [subset]
    # elif args[1] == "Calpha_iters_multiple_subset":
    #     system = args[2]
    #     task_name = args[3]
    #     dataset_name = args[4]
    #     begin_subset = int(args[5])
    #     end_subset = int(args[6])
    #     subsets = list(range(begin_subset, end_subset + 1))
    # for subset in subsets:
    #     data_path = "MLdata/" + system + "/" + dataset_name + "/subset_{:03d}.bin".format(subset)
    #     model_save_folder = "MLmodels/" + system + "/" + task_name + "/"
    #     cheatsheet_path = "MLdata/" + system + "/" + dataset_name + "/cheatsheet_set1_hankang.dat"
    #     print("tensorflow training info log")
    #     print("system", system, " task name", task_name, "subset", subset)
    #     print("loading data from ", data_path)
    #     print("loading cheatsheet from ", cheatsheet_path)
    #     print("save models to", model_save_folder)
    #     x, y = dm.get_data_integrated(data_size=0.4, begin_set=1, end_set=1, latter_cut_percentage=0.5,
    #                                   data_path=data_path, cheatsheet_path=cheatsheet_path, opt='equilibrium')
    #     print(x.shape)
    #     y = np.reshape(y, (len(y), 1))
    #     encoder = OneHotEncoder()
    #     encoder.fit(y)
    #     y = encoder.transform(y).toarray()
    #     print(y.shape)
    #     extra_info = {}
    #     dataV_path = "MLdata/" + system + "/" + dataset_name + "/subset_{:03d}_validation.bin".format(subset)
    #     cheatsheetV_path = "MLdata/" + system + "/" + dataset_name + "/cheatsheet_set1_validation.dat"
    #     x_V, y_V = dm.get_data_integrated(data_size=0.4, begin_set=1, end_set=1, latter_cut_percentage=0.5,
    #                                       data_path=dataV_path, cheatsheet_path=cheatsheetV_path, opt='equilibrium')
    #     y_V = np.reshape(y_V, (len(y_V), 1))
    #     encoder = OneHotEncoder()
    #     encoder.fit(y_V)
    #     y_V = encoder.transform(y_V).toarray()
    #     extra_info["x_V"] = x_V
    #     extra_info["y_V"] = y_V
    #     models_directory = model_save_folder + "models_subset_{:03d}/".format(subset)
    #     extra_info['directory'] = models_directory
    #     if not os.path.exists(models_directory):
    #         print("create model folder", models_directory)
    #         os.makedirs(models_directory, exist_ok=True)
    #     if len(os.listdir(models_directory)) >= 50:
    #         print("enough models in ", models_directory)
    #         continue
    #     for i in range(50 - len(os.listdir(models_directory))):
    #         x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    #         accuracy_train, accuracy_test, loss_list = tf_train(x_train, y_train, x_test, y_test, i, extra_info)
    #         # have a better model plot its accuracy and loss graph
    #         # if (accuracy_test[-1] > best_accuracy):
    #         #     best_accuracy = accuracy_test[-1]
    #         #     draw_figure(accuracy_train,accuracy_test,loss_list)
