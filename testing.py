from tensorflow import keras
import speck as sp




def evaluate(model, rounds):
    model = keras.models.load_model('./trained_networks/' + model)
    list_of_acc = []
    list_of_loss = []

    for x in range(rounds):
        X_test, Y_test = sp.make_train_data(10 ** 5, 6)
        loss, acc = model.evaluate(X_test, Y_test, batch_size=1000, verbose=0)
        list_of_acc.append(acc)
        list_of_loss.append(loss)

    loss_mean = 0
    for val in list_of_loss:
        loss_mean += val

    acc_mean = 0
    for val in list_of_acc:
        acc_mean += val

    acc_mean = float(acc_mean) / float(rounds)
    loss_mean = float(loss_mean) / float(rounds)
    print(model + ' loss_mean and acc_mean:')
    print(loss_mean)
    print(acc_mean)

    loss_SD = 0
    for val in list_of_loss:
        loss_SD += (float(val) - loss_mean) ** 2

    acc_SD = 0
    for val in list_of_acc:
        acc_SD += (float(val) - acc_mean) ** 2

    acc_SD = float(acc_SD) / float(rounds)
    loss_SD = float(loss_SD) / float(rounds)
    print(model + ' loss_SD and acc_SD:')
    print(loss_SD)
    print(acc_SD)


models = [  'conv_longresNet0.h5',
            'conv_longresNet2.h5',
            'conv_longresNet3.h5',
            'conv_longresNet.h5',
            'convNet.h5',
            'conv_resNet2.h5',
            'conv_resNet.h5']
for model in models:
    evaluate(model, 25)
