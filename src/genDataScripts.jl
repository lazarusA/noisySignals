using Random
using HDF5 # ProgressMeter
# τ >= 2  period and n sites
periodicSignal(n, τ) = [i % τ ==0 ? 1.0 : 0.0 for i in 0:n-1]

function signal_plus_noise(signal, σ)
    noise = σ .* abs.(randn(length(signal)))
    signalx =  signal .+ noise
    signalx./maximum(signalx)
end
function signal_plus_noise(signal, σ, doble)
    noise = σ .* randn(length(signal))
    signalx =  signal .+ noise
    minsig = minimum(signalx)
    signalx = signalx .- minsig
    signalx ./maximum(signalx)
end

function gen_samples_signal(signal, σ, num_samples)
    set_samples = zeros(length(signal), num_samples)
    for i in 1:num_samples
        set_samples[:, i] = signal_plus_noise(signal, σ)
    end
    set_samples
end

function gen_samples_signal(signal, σ, num_samples, doble)
    set_samples = zeros(length(signal), num_samples)
    for i in 1:num_samples
        set_samples[:, i] = signal_plus_noise(signal, σ, doble)
    end
    set_samples
end

function gen_data_set(all_signals, σ, num_samples)
    set_samples = gen_samples_signal(all_signals[1, :], σ, num_samples)
    labels = fill(1, num_samples)
    for i in 2:size(all_signals)[1]
        set_samplesx = gen_samples_signal(all_signals[i, :], σ, num_samples)
        set_samples = hcat(set_samples, set_samplesx)
        labelsx = fill(i, num_samples)
        labels = vcat(labels,labelsx)
    end
    set_samples, labels
end
function gen_data_set(all_signals, σ, num_samples, doble)
    set_samples = gen_samples_signal(all_signals[1, :], σ, num_samples, doble)
    labels = fill(1, num_samples)
    for i in 2:size(all_signals)[1]
        set_samplesx = gen_samples_signal(all_signals[i, :], σ, num_samples, doble)
        set_samples = hcat(set_samples, set_samplesx)
        labelsx = fill(i, num_samples)
        labels = vcat(labels,labelsx)
    end
    set_samples, labels
end

function gen_signals(n_sites, periods)
    all_signals = zeros(length(periods), n_sites)
    i = 1
    for τ in periods
        all_signals[i, :] = periodicSignal(n_sites, τ)
        i += 1
    end
    all_signals
end
