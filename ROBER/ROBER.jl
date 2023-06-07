using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots
using DiffEqSensitivity
using Zygote
using ForwardDiff
using LinearAlgebra
using Random
using Statistics
using ProgressBars, Printf
using Flux.Optimise: update!
using Flux.Losses: mae, mse
using BSON: @save, @load


dir         = @__DIR__
dir         = dir*"/"
cd(dir)
mkpath(dir*"figs")
mkpath(dir*"checkpoint")

is_restart  = false;
n_epoch     = 50000;
ntotal      = 50
n_plot      = 200;

batch_size  = 50;
opt         = ADAMW(0.005, (0.9, 0.999), 1.f-6);
lb          = 1.e-6;
ode_solver  = AutoTsit5(Rosenbrock23(autodiff=false));

function rober!(du, u, k, t)
    y1, y2, y3  = u
    k1, k2, k3  = k
    du[1]       = -k1 * y1 + k3 * y2 * y3
    du[2]       =  k1 * y1 - k3 * y2 * y3 - k2 * y2^2
    du[3]       =  k2 * y2^2
end

u0          = [1.0, 0, 0];
tspan       = (0.0, 1e5);
t_end       = tspan[2];
k           = [0.04, 3e7, 1e4];
tsteps      = 10 .^ (range(-5, log10(tspan[2]), length=ntotal));
prob_rober  = ODEProblem(rober!, u0, tspan, k);
sol_rober   = solve(prob_rober, ode_solver, saveat=tsteps, abstol=1.f-8);
normdata    = Array(sol_rober)

yscale      = maximum(normdata, dims = 2)  #scale for each species

i_slow      = [1, 2, 3];
nslow       = length(i_slow);
node        = 5;
dudt2       = Chain(x -> x,
                  Dense(nslow, node, gelu),
                  Dense(node, node, gelu),
                  Dense(node, node, gelu),
                  Dense(node, node, gelu),
                  Dense(node, node, gelu),
                  Dense(node, node, gelu),
                  Dense(node, nslow))

p, re       = Flux.destructure(dudt2);
rep         = re(p)

yscale_     = yscale[:, 1]
function dudt!(du, u, p, t)
    du      .= rep(u) .* yscale_ /t_end
end

prob        = ODEProblem(dudt!, u0[i_slow], tspan)
sense       = DiffEqSensitivity.BacksolveAdjoint(checkpointing=true; autojacvec=DiffEqSensitivity.ZygoteVJP());
function predict_n_ode(p, sample)
    global rep  = re(p)
    _prob       = remake(prob, p=p, tspan=[0, tsteps[sample]])
    pred        = Array(solve(_prob, ode_solver, saveat=tsteps[1:sample], abstol=lb, sensealg=sense))
end
pred        = predict_n_ode(p, ntotal)

function loss_n_ode(p, sample=ntotal)
    pred = predict_n_ode(p, sample)
    loss = mae(pred ./ yscale, normdata[i_slow, 1:size(pred)[2]] ./ yscale)
    return loss
end
loss_n_ode(p, ntotal)

list_loss = []
list_grad = []
iter = 1
cb = function (p, loss_mean, g_norm)
    global list_loss, list_grad, iter
    push!(list_loss, loss_mean)
    push!(list_grad, g_norm)

    if iter % n_plot == 0
        pred = predict_n_ode(p, ntotal)

        list_plt = []
        for i in 1:nslow
            j = i_slow[i]
            plt = scatter(tsteps[2:end], normdata[j,2:end], xscale=:log10, label="data")
            plot!(plt, tsteps[2:end], pred[i,2:end], lw=2, xscale=:log10, label="pred", framestyle=:box)
            plot!(plt, xtickfontsize=11, ytickfontsize=11, xguidefontsize=12, yguidefontsize=12)
            ylabel!(plt, "y$j")
            xlabel!(plt, "Time")
            if i == 1
                plot!(plt, legend=:best)
            else
                plot!(plt, legend=false)
            end
            push!(list_plt, plt)
        end
        plt_all = plot(list_plt..., layouts = (nslow, 1))
        png(plt_all, "figs/pred.png")

        plt_loss = plot(list_loss, xscale=:identity, yscale=:log10, label="loss", legend=:topright)
        plt_grad = plot(list_grad, xscale=:identity, yscale=:log10, label="grad_norm", legend=:bottomright)
        xlabel!(plt_loss, "Epoch")
        xlabel!(plt_grad, "Epoch")
        ylabel!(plt_loss, "Loss")
        ylabel!(plt_grad, "Gradient Norm")
        plt_all = plot([plt_loss, plt_grad]..., framestyle=:box, layout=(1,2))#, size = (1000, 400))
        plot!(plt_all, xtickfontsize=10, ytickfontsize=10, xguidefontsize=12, yguidefontsize=12)
        png(plt_all, "figs/loss_grad")

        @save "./checkpoint/mymodel.bson" p opt list_loss list_grad iter
    end
    iter += 1
    return false
end

if is_restart
    @load "./checkpoint/mymodel.bson" p opt list_loss list_grad iter
    iter += 1
end

epochs = ProgressBar(iter:n_epoch);
for epoch in epochs
    global p
    sample = rand(batch_size:ntotal)

    loss = loss_n_ode(p, sample)
    grad = ForwardDiff.gradient(x -> loss_n_ode(x, sample), p)

    grad_norm = norm(grad, 2)
    update!(opt, p, grad)

    set_description(epochs, string(@sprintf("Loss: %.4e grad: %.2e", loss, grad_norm)))
    cb(p, loss, grad_norm)
end
