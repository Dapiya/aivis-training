from tqdm import tqdm
import torch
import gc

def memClear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def inverse_transform(transformed_img):
    return 0.5 * (transformed_img + 1)

def train_one_epoch_with_accumulation(G, D, train_loader, writer, optim_G, optim_D, scaler_G, scaler_D,
                                      GAN_loss, lpips_loss, ssim_loss, image_size, batch_size, device,
                                      plot_every, epoch, epochs, use_amp=False, accumulation_steps=1):
    num_batches = len(train_loader)
    num_optimizer_steps = num_batches // accumulation_steps
    if num_batches % accumulation_steps != 0:
        num_optimizer_steps += 1

    pd = tqdm(total=num_optimizer_steps, desc='[Train %d/%d]' % (epoch + 1, epochs), unit='step')

    ls_D = ls_G = ls_ssim = ls_lpips = 0

    # Adversarial ground truths
    patch = (1, image_size // 2 ** 5, image_size // 2 ** 5)
    valid = torch.ones((batch_size, *patch)).to(device, non_blocking=True)
    fake = torch.zeros((batch_size, *patch)).to(device, non_blocking=True)
    
    if hasattr(torch, 'compile'):
        G = torch.compile(G, backend='eager')
        D = torch.compile(D, backend='eager')
        

    G.train()
    D.train()

    # Initialize counters
    optim_G.zero_grad()
    accumulation_counter = 0
    step = 0  # For plotting

    for idx, (in_img, real_img) in enumerate(train_loader):
        # Data Loading
        in_img = in_img.to(device, non_blocking=True)
        real_img = real_img.to(device, non_blocking=True)

        # ---------------------
        #  Train Generator
        # ---------------------
        with torch.cuda.amp.autocast(enabled=use_amp):
            fake_img = G(in_img)
            pred_fake = D(fake_img, in_img)
            loss_GAN = GAN_loss(pred_fake.float(), valid)
            loss_ssim = ssim_loss(
                inverse_transform(fake_img.float()),
                inverse_transform(real_img)
            )
            loss_lpips = lpips_loss(real_img, fake_img.float()).mean()
            loss_G = loss_lpips * 12 + (1 - loss_ssim) * 10 + 0.5 * loss_GAN

        # Accumulate generator loss
        ls_G += loss_G.item()
        ls_ssim += loss_ssim.item()
        ls_lpips += loss_lpips.item()

        # Scale loss for gradient accumulation
        loss_G_scaled = loss_G / accumulation_steps
        scaler_G.scale(loss_G_scaled).backward()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optim_D.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_fake = D(fake_img.detach(), in_img)
            pred_real = D(real_img, in_img)
            loss_fake = GAN_loss(pred_fake.float(), fake)
            loss_real = GAN_loss(pred_real.float(), valid)
            loss_D = (loss_fake + loss_real) * 0.5

        # Accumulate discriminator loss
        ls_D += loss_D.item()

        # Backward and optimizer step for Discriminator
        scaler_D.scale(loss_D).backward()
        scaler_D.step(optim_D)
        scaler_D.update()
        optim_D.zero_grad()

        accumulation_counter += 1

        if accumulation_counter % accumulation_steps == 0 or idx == num_batches - 1:
            # Optimizer step and zero_grad for Generator
            scaler_G.step(optim_G)
            scaler_G.update()
            optim_G.zero_grad()

            # Update progress bar
            pd.update(1)
            pd.set_postfix({
                'D loss': f'{loss_D.item():.8f}',
                'G loss': f'{loss_G.item():.8f}',
                'SSIM': f'{loss_ssim.item():.4f}',
                'LPIPS': f'{loss_lpips.item():.4f}',
                'adv': f'{loss_GAN.item():.8f}'
            })

            # Plot training results
            if plot_every > 0 and step % plot_every == 0:
                writer.add_images(tag='train_epoch_{}'.format(epoch), img_tensor=inverse_transform(fake_img), global_step=step)
            step += 1

    pd.close(); memClear()

    mean_lsG = ls_G / num_batches
    mean_lsD = ls_D / num_batches
    mean_SSIM = ls_ssim / num_batches
    mean_LPIPS = ls_lpips / num_batches

    return mean_lsG, mean_lsD, mean_SSIM, mean_LPIPS


@torch.no_grad()
def val_one_epoch(G, D, val_loader, writer, GAN_loss, ssim_loss, lpips_loss,
                  image_size, batch_size, device, epoch, epochs):
    pd = tqdm(val_loader)

    ls_D = ls_G = ls_ssim = ls_lpips = 0
    best_image = None
    all_loss = 10000

    # Adversarial ground truths
    patch = (1, image_size // 2 ** 5, image_size // 2 ** 5)
    valid = torch.ones((batch_size, *patch)).to(device, non_blocking=True)
    fake = torch.zeros((batch_size, *patch)).to(device, non_blocking=True)

    G.eval()
    D.eval()

    for idx, (in_img, real_img) in enumerate(pd):
        in_img = in_img.to(device, non_blocking=True)
        real_img = real_img.to(device, non_blocking=True)

        fake_img = G(in_img)
        pred_fake = D(fake_img, in_img)

        loss_fake = GAN_loss(pred_fake, fake)
        loss_D = loss_fake * 0.5

        loss_GAN = GAN_loss(pred_fake, valid)
        loss_lpips = lpips_loss.forward(real_img, fake_img).mean()
        loss_ssim = ssim_loss(
            inverse_transform(fake_img),
            inverse_transform(real_img)
        )
        loss_G = loss_lpips * 12 + (1 - loss_ssim) * 10 + loss_GAN
        
        ls_D += loss_D.item()
        ls_G += loss_G.item()
        ls_ssim += loss_ssim.item()
        ls_lpips += loss_lpips.item()

        # 输出训练信息
        pd.desc = '[Val %d/%d] [D loss: %.8f] [G loss: %.8f, SSIM: %.4f, LPIPS: %.4f, adv: %.8f]' % (
                    epoch + 1, epochs, loss_D.item(), loss_G.item(), loss_ssim.item(), loss_lpips.item(), loss_GAN.item())

        # 保存最好的结果
        all_ls = loss_G.item() + loss_D.item()
        if all_ls < all_loss:
            all_loss = all_ls
            best_image = inverse_transform(fake_img)

    pd.close(); memClear()

    mean_lsG = ls_G / len(val_loader)
    mean_lsD = ls_D / len(val_loader)
    mean_SSIM = ls_ssim / len(val_loader)
    mean_LPIPS = ls_lpips / len(val_loader)

    return mean_lsG, mean_lsD, mean_SSIM, mean_LPIPS, best_image
