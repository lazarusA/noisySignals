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
#trainSizes = [10000, 20000, 40000]
trainSizes = [100000]

# models are here
fpath = "/data/finite/lalonso/evaluations/"


sigmasAll = collect(0.05:0.05:4)
sigmasAllselect = [0.5,0.75,1,1.25,1.5,2]
indice = parse(Int64, ARGS[1])
#indice = 9
#σv = round(σvt*0.05, digits=3)
#σv = 2.5 # 1.0

grid = []
for ts in trainSizes, σv in sigmasAll
    push!(grid, [ts, σv])
end
tssize = Int64(grid[indice][1])
trainingSizes = [tssize]
σv = grid[indice][2]
iteraciones = 100

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
ycold9kxh = onehotbatch(ycold9kx, 1:10)
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

accuracyT(x, y, model) = mean(onecold(model(x)) .== onecold(y))
accuracybySample(x, y, model) = onecold(model(x)) .== onecold(y)

#t_size = 100_000

trainSizes = [100_000]
t_size=100_000

num_train = Int64(round.(t_size *0.8/10)) #number of samples for each class,
#this just works for 10 classes
num_val = Int64(round.(t_size *0.2/10))

xtrain_temp = x_train[:,1:num_train,:]
ytrain_temp = y_train[1:num_train,:]

xvali_temp = x_vali[:,1:num_val,:]
yvali_temp = y_vali[1:num_val,:]

x_val = reshape(xvali_temp, (n_sites, 10*num_val))
labels_val = reshape(yvali_temp, (10*num_val))

x_data = reshape(xtrain_temp, (n_sites, 10*num_train))
labels = reshape(ytrain_temp, (10*num_train))

X_val = x_val # |> gpu
Y_val = onehotbatch(labels_val, 1:10); #|> gpu

# Data preparation
X = x_data # |> gpu
Y = onehotbatch(labels, 1:10); #|> gpu

trainErrors = zeros(10,2,100)
valErrors = zeros(10,2,100)
testErrors = zeros(10,2,100)
testErrors_hhg = zeros(10,2,100,length(sigmasAllselect))
testErrors_s = zeros(10,2,100,length(sigmasAllselect))

for i in 1:100
    model = BSON.load(fpath*"modelsN100ESP_ts$(t_size)_σ$(σv)_ite$(i).bson")[:model]
    
    test_acc = accuracybySample(X_test, Y_test, model)
    train_acc = accuracybySample(X, Y, model)
    val_acc = accuracybySample(X_val, Y_val, model)
    
    train_fails = findall(iszero, train_acc*1)
    val_fails = findall(iszero, val_acc*1)
    test_fails = findall(iszero, test_acc*1)

    if length(train_fails) > 0
        samples_train = X[:,train_fails]
        pred_train = onecold(model(samples_train)) 
        true_train =  onecold(Y[:,train_fails])
        
        u1 = unique(pred_train)
        u2 = unique(true_train)
        d1 = Dict([(j, count(x->x==j,pred_train)) for j in u1])
        d2 = Dict([(j, count(x->x==j,true_train)) for j in u2])
        vals1 = hcat([[key, val] for (key, val) in d1]...)'
        vals2 = hcat([[key, val] for (key, val) in d2]...)'
        
        trainErrors[vals1[:,1],1,i] = vals1[:,2]
        trainErrors[vals2[:,1],2,i] = vals2[:,2]
        
    end
    if length(val_fails)>0
        samples_val = X_val[:,val_fails]
        pred_val = onecold(model(samples_val))
        true_val = onecold(Y[:,val_fails])
        u1 = unique(pred_val)
        u2 = unique(true_val)
        d1 = Dict([(j, count(x->x==j,pred_val)) for j in u1])
        d2 = Dict([(j, count(x->x==j,true_val)) for j in u2])
        vals1 = hcat([[key, val] for (key, val) in d1]...)'
        vals2 = hcat([[key, val] for (key, val) in d2]...)'
        
        valErrors[vals1[:,1],1,i] = vals1[:,2]
        valErrors[vals2[:,1],2,i] = vals2[:,2]
        
        
    end
    if length(test_fails)>0
        samples_test = X_test[:,test_fails]
        pred_test = onecold(model(samples_test))
        true_test = onecold(Y[:,test_fails])
        u1 = unique(pred_test)
        u2 = unique(true_test)
        d1 = Dict([(j, count(x->x==j,pred_test)) for j in u1])
        d2 = Dict([(j, count(x->x==j,true_test)) for j in u2])
        vals1 = hcat([[key, val] for (key, val) in d1]...)'
        vals2 = hcat([[key, val] for (key, val) in d2]...)'
        
        testErrors[vals1[:,1],1,i] = vals1[:,2]
        testErrors[vals2[:,1],2,i] = vals2[:,2]
        
    end
    
    for (indj, σvd) in enumerate(sigmasAllselect)
        signals_mas_ruido = signs100k(σvd)
        test_acc_hhg = accuracybySample(signals_mas_ruido, ycold9kxh, model)
        test_fails_hhg = findall(iszero, test_acc_hhg*1)
        
        if length(test_fails_hhg)>0
            samples_test = signals_mas_ruido[:,test_fails_hhg]
            pred_test = onecold(model(samples_test))
            true_test = onecold(ycold9kxh[:,test_fails_hhg])
            u1 = unique(pred_test)
            u2 = unique(true_test)
            d1 = Dict([(j, count(x->x==j, pred_test)) for j in u1])
            d2 = Dict([(j, count(x->x==j, true_test)) for j in u2])
            vals1 = hcat([[key, val] for (key, val) in d1]...)'
            vals2 = hcat([[key, val] for (key, val) in d2]...)'
            testErrors_hhg[vals1[:,1], 1, i, indj] = vals1[:,2]
            testErrors_hhg[vals2[:,1], 2, i, indj] = vals2[:,2]
            
        end
        
    end
    for (indj, σvd) in enumerate(sigmasAllselect)
        x_test_s, labels_test_s = synthData(σvd)
        labels_test_sh = onehotbatch(labels_test_s, 1:10)
        test_acc_s = accuracybySample(x_test_s, labels_test_sh, model)
        test_fails_s = findall(iszero, test_acc_s*1)
        if length(test_fails_s)>0
            samples_test = x_test_s[:,test_fails_s]
            pred_test = onecold(model(samples_test))
            true_test = onecold(labels_test_sh[:,test_fails_s])
            u1 = unique(pred_test)
            u2 = unique(true_test)
            d1 = Dict([(j, count(x->x==j, pred_test)) for j in u1])
            d2 = Dict([(j, count(x->x==j, true_test)) for j in u2])
            vals1 = hcat([[key, val] for (key, val) in d1]...)'
            vals2 = hcat([[key, val] for (key, val) in d2]...)'
            testErrors_s[vals1[:,1], 1, i, indj] = vals1[:,2]
            testErrors_s[vals2[:,1], 2, i, indj] = vals2[:,2]
            
        end
    end
    #println("iteration: $(i)")
end


npzwrite("./bump/bump_hhg_$(t_size)_sig_$(σv).npy", testErrors_hhg)
npzwrite("./bump/bump_test100k_$(t_size)_sig_$(σv).npy", testErrors_s)
npzwrite("./bump/bump_train_$(t_size)_sig_$(σv).npy", trainErrors)
npzwrite("./bump/bump_val_$(t_size)_sig_$(σv).npy", valErrors)
npzwrite("./bump/bump_test10k_$(t_size)_sig_$(σv).npy", testErrors)

 
