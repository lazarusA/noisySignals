#tbeg = time()
using LinearAlgebra
BLAS.set_num_threads(4)
using Flux, Statistics # Flux.Data.MNIST
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using BSON, HDF5, JLD, Random
using MLDataUtils
using NPZ
include("genDataScripts.jl")
n_sites = 100
periods = [2,3,4,5,6,7,8,9,10,11]
all_signals = gen_signals(n_sites, periods)

neu = 100
#trainingSizes = [50, 100, 250, 500, 1000, 2000, 4000]
trainSizes = [50, 100, 250, 500, 1000, 2000, 4000]
sigmasAll = collect(0.05:0.05:4)
indice = parse(Int64, ARGS[1])
#σv = round(σvt*0.05, digits=3)
#σv = 2.5 # 1.0

grid = []
for ts in trainSizes, σv in sigmasAll
    push!(grid, [ts, σv])
end
tssize = Int64(grid[indice][1])
trainingSizes = [tssize]
σv = grid[indice][2]
iteraciones = 50

# HHG DATA
x_data9k = npzread("./data/data_simulation2_optimized_9669.npy")
labels9k = npzread("./data/labels_simulation2_optimized_9669.npy")
ycold9k = Int64.(labels9k[:,1,1])
nsamples = size(x_data9k)[1]
npts = 100
data_log9k = zeros(npts, nsamples)
for i in 1:nsamples
    input_log = log10.(x_data9k[i,:,2])
    input_log = input_log .+ minimum(input_log)*(-1)
    #input_log = input_log/maximum(input_log)
    data_log9k[:,i] .= input_log
end
accuracy_tcold(x,y,model) = mean(onecold(model(x)) .== y)
function signs100k(σv)
    Random.seed!(1234)
    signals_mas_ruido = zeros(npts,nsamples)
    for i in 1:nsamples
        signals_mas_ruido[:,i] .= signal_plus_noise(data_log9k[:,i], σv, true)
    end
    for _ in 1:9
        signals_mas_ruidox = zeros(npts, nsamples)
        for i in 1:nsamples
            signals_mas_ruidox[:,i] .= signal_plus_noise(data_log9k[:,i], σv, true)
        end
        signals_mas_ruido = hcat(signals_mas_ruido, signals_mas_ruidox)
    end
    signals_mas_ruido
end
ycold9kx = repeat(ycold9k, 10)

# Synthetic data
function synthData(σv)
    all_signals = gen_signals(n_sites, periods)
    Random.seed!(1234)
    num_test = 10000
    x_test, y_test = gen_data_set(all_signals, σv, num_test,true)
    xtest_temp, ytest = reshape(x_test, (100, num_test, 10)), reshape(y_test, (num_test, 10))
    x_test = reshape(xtest_temp, (n_sites, 10*num_test))
    labels_test = reshape(ytest, (10*num_test))
    x_test, labels_test
end

Random.seed!(1234)
t_size = 1_000_000 # 4_000_000 old
num_train = Int64(round.(t_size *0.8/10)) # number of samples for each class,
#this just works for 10 classes
num_val = Int64(round.(t_size *0.2/10))
num_test = 1000

x_test, y_test = gen_data_set(all_signals, σv, num_test,true)
x_train, y_train = gen_data_set(all_signals, σv, num_train,true)
x_vali, y_vali = gen_data_set(all_signals, σv, num_val,true)

x_test = reshape(x_test, (100, num_test, 10))
y_test = reshape(y_test, (num_test, 10))
x_train = reshape(x_train, (100, num_train, 10))
y_train = reshape(y_train, (num_train, 10))
x_vali = reshape(x_vali, (100, num_val, 10))
y_vali = reshape(y_vali, (num_val, 10))

xtest_temp = x_test[:,1:num_test,:]
ytest_temp = y_test[1:num_test,:]
x_test = reshape(xtest_temp, (n_sites, 10*num_test))
labels_test = reshape(ytest_temp, (10*num_test))

X_test = x_test # |> gpu
Y_test = onehotbatch(labels_test, 1:10); #|> gpu
for t_size in trainingSizes
    tmtrain = time()
    num_train = Int64(round.(t_size *0.8/10)) #number of samples for each class,
    #this just works for 10 classes
    num_val = Int64(round.(t_size *0.2/10))

    xtrain_temp = x_train[:,1:num_train,:]
    ytrain_temp = y_train[1:num_train,:]

    x_data = reshape(xtrain_temp, (n_sites, 10*num_train))
    labels = reshape(ytrain_temp, (10*num_train))

    xvali_temp = x_vali[:,1:num_val,:]
    yvali_temp = y_vali[1:num_val,:]

    x_val = reshape(xvali_temp, (n_sites, 10*num_val))
    labels_val = reshape(yvali_temp, (10*num_val))


    # Data preparation
    X = x_data # |> gpu
    Y = onehotbatch(labels, 1:10); #|> gpu
    #dataset = [(X, Y)]
    X_val = x_val # |> gpu
    Y_val = onehotbatch(labels_val, 1:10); #|> gpu

    #mini-batch
    batch_size = 256 #512 is ok too, similar performance at test accuracy
    mb_idxs = partition(1:size(X)[2], batch_size)
    #train_set = [(X[:,p], Y[:,p]) for p in mb_idxs]
    for itera in 51:iteraciones+50
        # saving training history
        epochs = 1000
        t_loss = fill(NaN, epochs)
        v_loss = fill(NaN, epochs)
        va_acc = fill(NaN, epochs)

        #saving best models
        dict_models = Dict()

        #dict_models["model_iter"] = NaN
        dict_models["iter_epoch_num"] = 0


        #saving best model
        best_val = 5.0
        last_improvement = 0
        patience=50

        Random.seed!(10*itera)

        model = Chain(
          Dense(n_sites, neu, Flux.relu),
          Dense(neu, 10),
          Flux.softmax) # |> gpu
        opt = ADAM()
        loss(x, y) = crossentropy(model(x), y)
        global accuracyT(x, y) = mean(onecold(model(x)) .== onecold(y))

        for epoch_indx in 1:epochs
            xs, ys = shuffleobs((X, Y))
            train_set = [(xs[:,p], ys[:,p]) for p in mb_idxs]
            Flux.train!(loss, params(model), train_set, opt)

            validation_loss = Tracker.data(loss(X_val, Y_val))
            t_loss[epoch_indx] = Tracker.data(loss(X, Y))
            v_loss[epoch_indx] = validation_loss
            va_acc[epoch_indx] = accuracyT(X_val, Y_val)

            # If this is the best val_loss we've seen so far, save the model out
            if validation_loss <= best_val
                #@info(" -> New best val_loss! Saving model out to model_iter$(r).bson")
                last_improvement = epoch_indx
                dict_models["iter_epoch_num"] = last_improvement
                best_val = validation_loss
            end
            if epoch_indx - last_improvement >= patience
                #@info(" -> Early-exiting iteration $(r) and epoch $(epoch_indx): no more patience")
                break
            end
        end
        test_acc = accuracyT(X_test, Y_test)

        BSON.@save "modelsN$(neu)ESP_ts$(t_size)_σ$(σv)_ite$(itera).bson" model t_loss v_loss va_acc test_acc

        error_TS_ruido = zeros(length(sigmasAll))
        for (i,σvd) in enumerate(sigmasAll)
            signals_mas_ruido = signs100k(σvd)
            error_TS_ruido[i] = (1.0 .- accuracy_tcold(signals_mas_ruido, ycold9kx, model))
        end
        npzwrite("error_hhg_$(t_size)_1n_sig_$(σv)_ite$(itera).npy", error_TS_ruido)

        error_TS_ruidox = zeros(length(sigmasAll))
        for (i,σvd) in enumerate(sigmasAll)
            x_test, labels_test = synthData(σvd)
            error_TS_ruidox[i] = (1.0 .- accuracy_tcold(x_test, labels_test, model))
        end
        npzwrite("error_GtoG100k_$(t_size)_1n_sig_$(σv)_ite$(itera).npy", error_TS_ruidox)
    end
end
#println(time() - tbeg)
