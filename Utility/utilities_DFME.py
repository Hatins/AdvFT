import numpy as np
import torch
import torch.nn.functional as F
from Utility.utilities_AdvFT import DP_poison

def estimate_gradient_objective(args, victim_model, clone_model, x, epsilon=1e-7, m=5, verb=False, num_classes=10,
                                device="cpu", pre_x=False):
    # Sampling from unit sphere is the method 3 from this website:
    #  http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    # x = torch.Tensor(np.arange(2*1*7*7).reshape(-1, 1, 7, 7))

    if pre_x and args.G_activation is None:
        raise ValueError(args.G_activation)

    clone_model.eval()
    victim_model.eval()
    with torch.no_grad():
        # Sample unit noise vector
        N = x.size(0)
        C = x.size(1)
        S = x.size(2)
        dim = S ** 2 * C

        u = np.random.randn(N * m * dim).reshape(-1, m, dim)  # generate random points from normal distribution
        d = np.sqrt(np.sum(u ** 2, axis=2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
        u = torch.Tensor(u / d).view(-1, m, C, S, S)
        u = torch.cat((u, torch.zeros(N, 1, C, S, S)), dim=1)  # Shape N, m + 1, S^2
        u = u.view(-1, m + 1, C, S, S)

        evaluation_points = (x.view(-1, 1, C, S, S).cpu() + epsilon * u).view(-1, C, S, S)
        if pre_x:
            evaluation_points = args.G_activation(evaluation_points)  # Apply args.G_activation function

        # Compute the approximation sequentially to allow large values of m
        pred_victim = []
        pred_clone = []
        max_number_points = 32 * 156  # Hardcoded value to split the large evaluation_points tensor to fit in GPU

        for i in (range(N * m // max_number_points + 1)):
            pts = evaluation_points[i * max_number_points: (i + 1) * max_number_points]
            pts = pts.to(device)
            _, pred_victim_pts = victim_model(pts)
            pred_victim_pts = pred_victim_pts.detach()
            _, pred_clone_pts = clone_model(pts)

            pred_victim.append(pred_victim_pts)
            pred_clone.append(pred_clone_pts)
        pred_victim = torch.cat(pred_victim, dim=0).to(device)
        pred_clone = torch.cat(pred_clone, dim=0).to(device)

        u = u.to(device)

        if args.loss == "l1":
            loss_fn = F.l1_loss
            if args.no_logits:
                pred_victim = F.log_softmax(pred_victim, dim=1).detach()
                if args.logit_correction == 'min':
                    pred_victim -= pred_victim.min(dim=1).values.view(-1, 1).detach()
                elif args.logit_correction == 'mean':
                    pred_victim -= pred_victim.mean(dim=1).view(-1, 1).detach()

        elif args.loss == "kl":
            loss_fn = F.kl_div
            pred_clone = F.log_softmax(pred_clone, dim=1)
            pred_victim = F.softmax(pred_victim.detach(), dim=1)

        else:
            raise ValueError(args.loss)

        # Compute loss
        if args.loss == "kl":
            loss_values = - loss_fn(pred_clone, pred_victim, reduction='none').sum(dim=1).view(-1, m + 1)
        else:
            loss_values = - loss_fn(pred_clone, pred_victim, reduction='none').mean(dim=1).view(-1, m + 1) \
                          # - torch.exp(energy).view(-1,m+1)


            # Compute difference following each direction
        differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
        differences = differences.view(-1, m, 1, 1, 1)
        # Formula for Forward Finite Differences
        gradient_estimates = 1 / epsilon * differences * u[:, :-1]

        if args.forward_differences:
            gradient_estimates *= dim

        if args.loss == "kl":
            gradient_estimates = gradient_estimates.mean(dim=1).view(-1, C, S, S)
        else:
            gradient_estimates = gradient_estimates.mean(dim=1).view(-1, C, S, S) / (num_classes * N)

        clone_model.train()
        loss_G = loss_values[:, -1].mean()
        return gradient_estimates.detach(), loss_G

def compute_gradient(args, victim_model, clone_model, x, pre_x=False, device="cpu"):
    if pre_x and args.G_activation is None:
        raise ValueError(args.G_activation)

    clone_model.eval()
    N = x.size(0)
    x_copy = x.clone().detach().requires_grad_(True)
    x_ = x_copy.to(device)

    if pre_x:
        x_ = args.G_activation(x_)

    _, pred_victim = victim_model(x_)
    _, pred_clone = clone_model(x_)

    if args.loss == "l1":
        loss_fn = F.l1_loss
        if args.no_logits:
            pred_victim_no_logits = F.log_softmax(pred_victim, dim=1)
            if args.logit_correction == 'min':
                pred_victim = pred_victim_no_logits - pred_victim_no_logits.min(dim=1).values.view(-1, 1)
            elif args.logit_correction == 'mean':
                pred_victim = pred_victim_no_logits - pred_victim_no_logits.mean(dim=1).view(-1, 1)
            else:
                pred_victim = pred_victim_no_logits

    elif args.loss == "kl":
        loss_fn = F.kl_div
        pred_clone = F.log_softmax(pred_clone, dim=1)
        pred_victim = F.softmax(pred_victim, dim=1)

    else:
        raise ValueError(args.loss)

    loss_values = -loss_fn(pred_clone, pred_victim, reduction='mean')
    # print("True mean loss", loss_values)
    loss_values.backward()

    clone_model.train()

    return x_copy.grad, loss_values

def student_loss(args, s_logit, t_logit, return_t_logits=False):
    """Kl/ L1 Loss for student"""
    print_logits =  False
    if args.loss == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif args.loss == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(args.loss)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss

def measure_true_grad_norm(args, x):
    # Compute true gradient of loss wrt x
    true_grad, _ = compute_gradient(args, args.Victim_model, args.Surrogate_model, x, pre_x=True, device=args.device)
    true_grad_clone = true_grad.clone()
    true_grad = true_grad.view(-1, 3*args.size*args.size)

    # Compute norm of gradients
    norm_grad = true_grad.norm(2, dim=1).mean().cpu()
    return true_grad_clone, norm_grad

def a_test(args, Surrogate_model = None, Generator = None, device = "cuda:0", test_loader = None, epoch=0):
    global file
    Surrogate_model.eval()
    Generator.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            _, output = Surrogate_model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    acc = correct/len(test_loader.dataset)
    return acc

def train(args, Victim_model, Surrogate_model,Generator,optimizer,epoch,device):
    Victim_model.eval()
    Surrogate_model.eval()

    optimizer_S, optimizer_G = optimizer
    for i in range(args.epoch_itrs):
        for _ in range(args.g_iter):
            noise = torch.randn((args.batch, args.noise_size)).to(device)
            optimizer_G.zero_grad() #set gradient to 0 in each epoch
            Generator.train()

            fake = Generator(noise, pre_x=args.approx_grad)
            approx_grad_wrt_x, loss_G = estimate_gradient_objective(args, Victim_model, Surrogate_model, fake,
                                                                    epsilon=args.grad_epsilon, m=args.grad_m,
                                                                    num_classes=args.num_classes,
                                                                    device=device, pre_x=True)
            fake.backward(approx_grad_wrt_x)
            optimizer_G.step()
        for _ in range(args.d_iter):
            noise = torch.randn((args.batch, args.noise_size)).to(device)
            fake = Generator(noise).detach()
            optimizer_S.zero_grad()
            with torch.no_grad():
                _, t_logit = Victim_model(fake)
                if args.poison == True:
                    t_soft = F.softmax(t_logit,dim=1).detach()
                    t_soft_poison = DP_poison(args,t_soft)
                    t_logit = torch.log(t_soft_poison)
                else:
                    t_logit = F.log_softmax(t_logit, dim=1).detach()

                if args.logit_correction == 'min':
                    t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
                elif args.logit_correction == 'mean':
                    t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()

            _, s_logit = Surrogate_model(fake)
            loss_S = student_loss(args, s_logit, t_logit)
            loss_S.backward()
            optimizer_S.step()

        if i % 10 == 0:
            print(
                f'Train Epoch: {epoch} [{i}/{args.epoch_itrs} ({100 * float(i) / float(args.epoch_itrs):.0f}%)]\tG_Loss: {loss_G.item():.6f} S_loss: {loss_S.item():.6f}')

        # update query budget
        args.query_times -= args.cost_per_iteration
