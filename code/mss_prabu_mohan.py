import os

from utils import DataLoader
from data_handling_prabu_mohan import MyDataset
from dae_prabu_mohan import MyModel
import numpy as np
import torch
from copy import deepcopy
from exercise_06 import to_audio
import librosa as lb
import soundfile as sf
import mir_eval


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    feature_root_dir = '../data/features/'
    mix_train_dir = feature_root_dir + 'mix/train/'
    mix_test_dir = feature_root_dir + 'mix/test/'
    source_train_dir = feature_root_dir + 'source/train/'
    source_test_dir = feature_root_dir + 'source/test/'
    batch_size=30
    mix_files=os.listdir(mix_train_dir)
    percent=int(0.9*len(mix_files))
    train_files=mix_files[:percent]
    val_files=mix_files[percent:]
    test_files=os.listdir(mix_test_dir)
    test1_files=[]
    test2_files=[]
    audio_root_dir = '../data/raw_data/'
    sources_dir = audio_root_dir + 'Sources/'
    mix_dir = audio_root_dir + 'Mixtures/'

    for i in test_files:
        if i.__contains__("testing1"):
            test1_files.append(i)
        else:
            test2_files.append(i)


    train_data_loader = DataLoader(
        MyDataset(train_files,mix_train_dir, source_train_dir, mix_prefix='mix_train_', source_prefix='sou_train_'), batch_size=batch_size)

    val_data_loader = DataLoader(
        MyDataset(val_files, mix_train_dir, source_train_dir, mix_prefix='mix_train_', source_prefix='sou_train_'),
        batch_size=batch_size)

    test1_data_loader = DataLoader(
        MyDataset(test1_files, mix_test_dir, source_test_dir, mix_prefix='mix_test_', source_prefix='sou_test_',test_file_path=mix_dir+"testing/testing_1/mixture.wav"),
        batch_size=batch_size,)
    test2_data_loader = DataLoader(
        MyDataset(test2_files, mix_test_dir, source_test_dir, mix_prefix='mix_test_', source_prefix='sou_test_',
                  test_file_path=mix_dir + "testing/testing_2/mixture.wav"),
        batch_size=batch_size, )

    gru_props = [(1025, 128, 1), (256, 128, 2), (256, 128, 1)]
    lin_props = [(128, 1025)]
    testing = MyModel(
        gru_props=gru_props,
        lin_props=lin_props
    ).to(device=device).double()
    rand_input=torch.rand((30,60,1025)).to(device).double()
    test_output=testing(rand_input)
    print("testing output shape "+str(test_output.shape))

    model = MyModel(
        gru_props=gru_props,
        lin_props=lin_props
    ).to(device=device).double()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    # Variables for the early stopping
    epoch = 200
    epoch=0
    lowest_val_loss = np.inf
    best_val_epoch = 0
    patience = 20
    patience_counter = 0
    best_model = None
    loss_function = torch.nn.MSELoss()
    model.train()
    for j in range(epoch):
        train_loss = []
        val_loss = []
        model.train()
        for i, (mix,source) in enumerate(train_data_loader):
            source=torch.abs(source)
            mix=torch.abs(mix)
            mix=mix.to(device=device).double()
            source=source.to(device=device).double()
            predict=model(mix)
            # print(predict.dtype)
            # print(source.dtype)
            # print(predict.shape)
            # print(source.shape)
            loss=loss_function(predict, source)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss)

        model.eval()
        with torch.no_grad():
                for i, (x, y) in enumerate(val_data_loader):
                    x = torch.abs(x).to(device).double()
                    y = torch.abs(y).to(device=device).double()
                    score = model(x)
                    loss = loss_function(score, y)
                    # print("val loss "+str( loss))
                    val_loss.append(loss)

        train_mean = torch.mean(torch.tensor(train_loss))
        val_mean = torch.mean(torch.tensor(val_loss))
        print("epoch ------ (" + str(j) + "/" + str(epoch - 1) + ") training loss---- " + str(
            train_mean) + "validation loss---- " + str(val_mean))
        if lowest_val_loss > val_mean:

            lowest_val_loss = val_mean
            print("new lowest val " + str(lowest_val_loss))
            best_val_epoch = j
            best_model = deepcopy(model.state_dict())
            torch.save(best_model, 'best_model')
            patience_counter = 0
        else:
            if patience < patience_counter:
                print("breaking the epoch loop as there is no improvement in the loss")
                break
            else:
                patience_counter = patience_counter + 1
    model = MyModel(
        gru_props=gru_props,
        lin_props=lin_props
    ).to(device=device).double()
    if best_model is None:
        best_model=torch.load('best_model')
    model.load_state_dict(best_model)
    model.eval()
    t_out=[]

    for i,(mix,raw) in enumerate(test1_data_loader):
        mix=mix.to(device=device).double()
        raw_data,sr=raw
        raw_data= predict=raw_data.cpu().detach().numpy().reshape(-1)
        print(sr)
        test_out=model(mix)
        print(test_out.shape)
        predict=test_out.cpu().detach().numpy()
        predict=np.reshape(predict,(predict.shape[0],predict.shape[2],predict.shape[1]))
        print(predict.shape)
        audio=to_audio(raw_data,predict)
        sf.write('../data/test_audio_1.wav',audio,sr)

    for i,(mix,raw) in enumerate(test2_data_loader):
        mix=mix.to(device=device).double()
        raw_data,sr=raw
        raw_data= predict=raw_data.cpu().detach().numpy().reshape(-1)
        print(sr)
        test_out=model(mix)
        print(test_out.shape)
        predict=test_out.cpu().detach().numpy()
        predict=np.reshape(predict,(predict.shape[0],predict.shape[2],predict.shape[1]))
        print(predict.shape)
        audio=to_audio(raw_data,predict)
        sf.write('../data/test_audio_2.wav',audio,sr)

    '''
    evaluation
    '''
    source_test1,sr=lb.load(sources_dir + 'testing/testing_1/vocals.wav')
    predicted_test1, sr = lb.load('../data/test_audio_1.wav')
    small_len= source_test1.size if source_test1.size< predicted_test1.size else predicted_test1.size
    (sdr, isr, sir, sar, perm)=mir_eval.separation.bss_eval_images_framewise(source_test1[:small_len] ,predicted_test1[:small_len] , window=sr, hop=sr, compute_permutation=False)
    print("test1 evaluation metrics")
    print(sdr)
    print(sir)
    print(sar)

    source_test1, sr = lb.load(sources_dir + 'testing/testing_2/vocals.wav')
    predicted_test1, sr = lb.load('../data/test_audio_2.wav')
    small_len = source_test1.size if source_test1.size < predicted_test1.size else predicted_test1.size
    (sdr, isr, sir, sar, perm) = mir_eval.separation.bss_eval_images_framewise(source_test1[:small_len],
                                                                               predicted_test1[:small_len], window=sr,
                                                                               hop=sr, compute_permutation=False)
    print("test2 evaluation metrics")
    print(sdr)
    print(sir)
    print(sar)
'''
test 1 evaluation metrics outputs
sdr ->
[[         nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan -40.01129477  -3.83108329  -1.23888506
   -1.4282664   -1.79978621  -1.55963868  -1.48087695  -1.58024986
   -1.88554576  -1.12958673  -1.24788105  -1.08009122  -1.2109985
   -0.90238636  -1.47659274  -1.16658218  -1.11671953  -7.9755793
  -39.03438099 -39.35937823 -38.93168171 -39.68960409 -40.5491172
  -38.86222111  -2.20003244  -1.65321735  -1.34535492  -2.28392714
   -1.74416813  -1.87001381  -1.58183771  -2.05694878  -1.1596491
   -1.26896075  -1.1125379   -1.19834898  -1.04806132  -1.50529258
   -1.56175303  -1.38140881 -29.82144069 -48.15319639 -37.90011128
   -2.04288439  -0.84674852  -1.02719892 -30.20148327  -1.19821972
   -0.9587477   -1.36417766  -1.66996983  -1.57461845  -1.01108253
   -1.06959724  -0.8447171   -0.73956495  -0.72986072 -17.24720249
  -37.13073769 -39.5148853  -36.07302888 -39.24527285 -36.79671116
  -40.70000801 -40.18976607 -18.39085328  -1.8635766   -2.61423622
   -1.28942656  -3.05715009  -2.24206791  -1.94120738  -1.86224868
   -1.26031773  -1.23843597  -1.10375657  -1.89316282  -0.8988565
   -1.21048165  -1.46423983  -1.26960071  -2.24987378 -40.85617003
  -49.42036981 -38.96687137  -1.5053182   -1.08766454  -3.62697046
   -5.03389416  -1.23682799  -1.09134634  -8.39020173  -1.53446288
   -1.3303331   -1.81413712  -1.23096177  -0.89352829  -1.06284694
   -1.20893042  -5.34997796  -0.85229355  -1.11296102  -1.77067283
  -25.06509543  -0.98096748  -0.71369704  -5.78656766  -1.34013125
   -1.04556411  -1.22670362  -1.05382     -0.95307101  -0.83324967
   -0.92346196 -27.23739487 -40.98470926 -48.37614213 -37.85899846
  -32.97283306 -35.35073001 -39.64652875 -41.86876469 -41.38538991
  -44.39054074 -44.22964724 -42.1990609  -42.75962931 -46.4510596
  -45.07898925          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan]]

SIR->
[[nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan inf
  inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
  inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
  inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
  inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
  inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
  inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
  inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf
  inf inf nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan
  nan nan nan nan nan nan nan nan]]
  
  SAR->
  [[         nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan -16.27488636 -13.6695987   -9.92178372
   -9.12237784  -9.77284694  -8.95550147  -9.89750088 -10.33580669
  -10.87428477 -10.40740162 -10.0175014  -10.67517173 -11.3928633
   -9.23642598  -9.14041561  -8.93380043 -10.70415632 -15.54539485
  -16.04358017 -16.18011228 -15.65610533 -15.96004856 -16.26930984
  -16.49709912 -11.94603519 -10.64482168 -10.51418556  -9.60842117
  -10.65577359  -9.83620952  -9.6956878   -9.25857321 -10.42813038
  -10.48721899 -10.04765958 -11.28323343 -10.30072633  -9.7296043
  -11.07253015 -12.4223206  -16.20612869 -16.37612689 -15.92193016
  -14.54189818 -12.0339334  -12.57969304 -15.7103813  -12.22857129
  -12.39768724 -12.87159246 -13.39362431 -10.69124744 -11.99941813
  -11.882386   -10.75879083 -11.69416864 -10.79570547 -15.51462489
  -16.04111666 -16.3754976  -15.89617646 -16.33344558 -16.25595895
  -16.32331575 -16.27692297 -17.09336564 -11.43227     -9.99941188
   -9.81320996 -10.76873863 -11.0418348  -11.08913331  -7.54705223
   -8.50376143 -10.6645493  -11.67221334 -11.2538919  -11.383087
   -8.91468407  -9.97189992 -11.38315027 -12.7726967  -15.63400287
  -16.2313412  -16.50306536 -12.33470759 -12.72495029 -13.84841426
  -15.46265766 -12.41781725 -11.84258948 -17.45970064 -11.70636108
  -12.11851857 -12.60485803 -11.95159872 -11.31428942 -11.5287944
  -12.67386035 -17.5243209  -11.48956108 -12.18471329 -13.56781576
  -20.27416541 -11.91710568 -11.55930491 -14.5063954  -11.29963835
  -11.39901599 -11.2971466  -10.97145717 -11.00083544 -10.63636291
  -12.03233209 -16.28139964 -15.48285867 -15.95355101 -16.17293074
  -16.27824575 -15.49747294 -16.06535662 -16.18777767 -16.2462307
  -15.67715085 -16.36201102 -15.85944677 -16.50358333 -16.2693608
  -15.21883148          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan          nan
           nan          nan          nan          nan]]
'''
if __name__ == '__main__':
    main()
