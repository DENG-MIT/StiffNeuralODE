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


is_restart = false
n_epoch = 25000;
ntotal = 20
n_plot = 50;
grad_max = 1.e2;
batch_size = ntotal;


opt = ADAMW(0.005, (0.9, 0.999), 1.f-6);


lb = 1.e-6;
ub = 1.e5;
ode_solver = AutoTsit5(Rosenbrock23(autodiff=false));


k1 = .35e0
k2 = .266e2
k3 = .123e5
k4 = .86e-3
k5 = .82e-3
k6 = .15e5
k7 = .13e-3
k8 = .24e5
k9 = .165e5
k10 = .9e4
k11 = .22e-1
k12 = .12e5
k13 = .188e1
k14 = .163e5
k15 = .48e7
k16 = .35e-3
k17 = .175e-1
k18 = .1e9
k19 = .444e12
k20 = .124e4
k21 = .21e1
k22 = .578e1
k23 = .474e-1
k24 = .178e4
k25 = .312e1

function pollu!(dy, y, p, t)
    r1  = k1 * y[1]
    r2  = k2 * y[2] * y[4]
    r3  = k3 * y[5] * y[2]
    r4  = k4 * y[7]
    r5  = k5 * y[7]
    r6  = k6 * y[7] * y[6]
    r7  = k7 * y[9]
    r8  = k8 * y[9] * y[6]
    r9  = k9 * y[11] * y[2]
    r10 = k10 * y[11] * y[1]
    r11 = k11 * y[13]
    r12 = k12 * y[10] * y[2]
    r13 = k13 * y[14]
    r14 = k14 * y[1] * y[6]
    r15 = k15 * y[3]
    r16 = k16 * y[4]
    r17 = k17 * y[4]
    r18 = k18 * y[16]
    r19 = k19 * y[16]
    r20 = k20 * y[17] * y[6]
    r21 = k21 * y[19]
    r22 = k22 * y[19]
    r23 = k23 * y[1] * y[4]
    r24 = k24 * y[19] * y[1]
    r25 = k25 * y[20]

    dy[1]  = -r1 - r10 - r14 - r23 - r24 + r2 + r3 + r9 + r11 + r12 + r22 + r25
    dy[2]  = -r2 - r3 - r9 - r12 + r1 + r21
    dy[3]  = -r15 + r1 + r17 + r19 + r22
    dy[4]  = -r2 - r16 - r17 - r23 + r15
    dy[5]  = -r3 + r4 + r4 + r6 + r7 + r13 + r20
    dy[6]  = -r6 - r8 - r14 - r20 + r3 + r18 + r18
    dy[7]  = -r4 - r5 - r6 + r13
    dy[8]  = r4 + r5 + r6 + r7
    dy[9]  = -r7 - r8
    dy[10] = -r12 + r7 + r9
    dy[11] = -r9 - r10 + r8 + r11
    dy[12] = r9
    dy[13] = -r11 + r10
    dy[14] = -r13 + r12
    dy[15] = r14
    dy[16] = -r18 - r19 + r16
    dy[17] = -r20
    dy[18] = r20
    dy[19] = -r21 - r22 - r24 + r23 + r25
    dy[20] = -r25 + r24
end


u0 = zeros(20);
u0[2]  = 0.2;
u0[4]  = 0.04;
u0[7]  = 0.1;
u0[8]  = 0.3;
u0[9]  = 0.01;
u0[17] = 0.007;


tspan = (0.0, 60.0);
t_end = tspan[2];
k = [0.04, 3e7, 1e4];
tsteps = range(0, t_end, length=ntotal);
prob_rober = ODEProblem(pollu!, u0, tspan, k);
sol_rober = solve(prob_rober, ode_solver, saveat=tsteps, atol=1.f-6, rtol=1e-12);
normdata = Array(sol_rober)


i_slow = 1:20
nslow = length(i_slow)
yscale = maximum(normdata, dims=2)[i_slow] - minimum(normdata, dims=2)[i_slow]


nslow = length(i_slow);
node = 10;
dudt2 = Chain(x -> x,
              Dense(nslow, node, gelu),
              Dense(node, node, gelu),
              Dense(node, node, gelu),
              Dense(node, nslow))

p, re = Flux.destructure(dudt2);
rep = re(p)

function dudt!(du, u, p, t)
    du .= rep(u) .* yscale / t_end
end

prob = ODEProblem(dudt!, u0[i_slow], tspan)
sense = BacksolveAdjoint(checkpointing=true; autojacvec=ZygoteVJP());


function predict_n_ode(p, sample)
    global rep = re(p)
    _prob = remake(prob, p=p, tspan=[0, tsteps[sample]])
    pred = Array(solve(_prob, ode_solver, saveat=tsteps[1:sample], atol=lb, sensalg=sense))
end
pred = predict_n_ode(p, ntotal)


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
            plt = scatter(tsteps[:], normdata[j,:], xscale=:identity, label="data")
            plot!(plt, tsteps[:], pred[i,:], lw=2, xscale=:identity, label="pred", framestyle=:box)
            if i == 1
                plot!(plt, legend=:best)
            else
                plot!(plt, legend=false)
            end
            push!(list_plt, plt)
        end
        plt_all = plot(list_plt..., size=(1500, 1500))
        png(plt_all, "figs/pred.png")

        plt_loss = plot(list_loss, xscale=:identity, yscale=:log10, label="loss", legend=:topright)
        plt_grad = plot(list_grad, xscale=:identity, yscale=:log10, label="grad_norm", legend=:bottomright)
        xlabel!(plt_loss, "Epoch")
        xlabel!(plt_grad, "Epoch")
        ylabel!(plt_loss, "Loss")
        ylabel!(plt_grad, "Gradient Norm")
        plt_all = plot([plt_loss, plt_grad]..., framestyle=:box, size=(1000, 400))
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
    grad = grad ./ grad_norm .* grad_max

    update!(opt, p, grad)

    set_description(epochs, string(@sprintf("Loss: %.4e grad: %.2e", loss, grad_norm)))
    cb(p, loss, grad_norm)
end
