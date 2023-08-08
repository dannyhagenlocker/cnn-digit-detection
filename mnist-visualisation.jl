
using Flux
using MLDatasets
using Plots

foo = "lenetmodel.bson"  # sub-directory in which to save

train_data = MLDatasets.MNIST()  # i.e. split=:train
test_data = MLDatasets.MNIST(split=:test)

function loader(data::MNIST=train_data; batchsize::Int=64)
    x4dim = reshape(data.features, 28,28,1,:)   # insert trivial channel dim
    yhot = Flux.onehotbatch(data.targets, 0:9)  # make a 10×60000 OneHotMatrix
    Flux.DataLoader((x4dim, yhot); batchsize, shuffle=true)
end

model = Chain(
    Conv((5, 5), 1=>6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6=>16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, relu),
    Dense(120 => 84, relu), 
    Dense(84 => 10),
)
using BSON: @load
using ImageCore, ImageInTerminal, Images, ImageMetadata, TestImages, ImageView, Zygote
@load "lenetmodel.bson" weights
@load "trainlog.bson" train_log
Flux.loadparams!(model, weights)

xtest, ytest = first(loader(test_data, batchsize=10));

@show Plots.plot(train_log)
Plots.scatter!(train_log)

#import Pkg
#Pkg.add("Zygote")


#for i in 1:6
#   save(string("layer1_channel_",i,".jpg"), colorview(Gray, model[1](xtest)[:,:,i,1]))
#    save(string("pool1_channel_",i,".jpg"), colorview(Gray, model[2](model[1](xtest))[:,:,i,1]))
#end
#for i in 1:16
#    save(string("layer2_channel_",i,".jpg"), colorview(Gray, model[3](model[2](model[1](xtest)))[:,:,i,1]))
#    save(string("pool2_channel_",i,".jpg"), colorview(Gray, model[4](model[3](model[2](model[1](xtest))))[:,:,i,1]))
#end
#save("preconvolution.jpg", colorview(Gray, xtest[:,:,1,1]))
#@show model(xtest)
#@show typeof(xtest)
#@show typeof(model[1](xtest))




#@show pred = Zygote.data(model(xtest)); 
