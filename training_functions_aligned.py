import torch
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from reporting import get_experiment_name
from torch import optim


def latent_loss(mu, log_var):
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    return kld_loss


def train_aligned(model, dataloader, num_epochs, criterion, optimizer, output_dir):
    model.decoder_type = model.decoder_type + "aligned"
    name_logging, name_model, name_writer, name = get_experiment_name(
        model=model, output_dir=output_dir
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.basicConfig(filename=name_logging, level=logging.INFO)
    logging.info(f"Started training model {name} at {now}.")
    writer = SummaryWriter(log_dir=name_writer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    best_loss = 1000000000
    niter = 1
    for epoch in range(num_epochs):
        batch_num = 1
        running_loss = 0.0
        model.train()
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs = data[1]
                inputs = inputs.to(device)
                # rotated_inputs = data[2]
                # rotated_inputs = roated_inputs.to(device)
                batch_size = inputs.shape[0]
                # ===================forward=====================
                with torch.set_grad_enabled(True):
                    output, features = model(inputs)
                    optimizer.zero_grad()
                    loss = criterion(output, inputs)
                    # ===================backward====================
                    loss.backward()
                    optimizer.step()

                batch_loss = loss.detach().item() / batch_size
                running_loss += batch_loss
                batch_num += 1
                writer.add_scalar("/Loss", batch_loss, niter)
                niter += 1
                tepoch.set_postfix(loss=batch_loss)

                if batch_num % 10 == 0:
                    logging.info(
                        f"[{epoch}/{num_epochs}]"
                        f"[{batch_num}/{len(dataloader)}]"
                        f"LossTot: {batch_loss}"
                    )

            total_loss = running_loss / len(dataloader)
            if total_loss < best_loss:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": total_loss,
                }
                best_loss = total_loss
                torch.save(checkpoint, name_model)
                logging.info(f"Saving model to {name_model} with loss = {best_loss}.")

        scheduler.step()

def train_vae(model, dataloader, num_epochs, criterion, optimizer, output_dir):

    name_logging, name_model, name_writer, name = get_experiment_name(
        model=model, output_dir=output_dir
    )

    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.basicConfig(filename=name_logging, level=logging.INFO)
    logging.info(f"Started training model {name} at {now}.")
    writer = SummaryWriter(log_dir=name_writer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    best_loss = 1000000000
    niter = 1
    for epoch in range(num_epochs):
        batch_num = 1
        running_loss = 0.0
        model.train()
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs = data[0]
                inputs = inputs.to(device)
                batch_size = inputs.shape[0]

                # ===================forward=====================
                with torch.set_grad_enabled(True):
                    output, mu, log_var, features, z = model(inputs)
                    optimizer.zero_grad()
                    loss_rec = criterion(inputs, output)
                    loss_kl = latent_loss(mu, log_var)
                    # ===================backward====================
                    loss = loss_rec + loss_kl
                    loss.backward()
                    optimizer.step()

                batch_loss = loss.detach().item() / batch_size
                batch_loss_rec = loss_rec.detach().item() / batch_size
                batch_loss_kl = loss_kl.detach().item() / batch_size
                running_loss += batch_loss
                batch_num += 1
                writer.add_scalar("/Loss", batch_loss, niter)
                niter += 1
                tepoch.set_postfix(loss=batch_loss)

                if batch_num % 10 == 0:
                    logging.info(
                        f"[{epoch}/{num_epochs}]"
                        f"[{batch_num}/{len(dataloader)}]"
                        f"LossTot: {batch_loss}"
                        f"Loss Rec: {batch_loss_rec}"
                        f"Loss kl: {batch_loss_kl}"
                    )
            total_loss = running_loss / len(dataloader)
            if total_loss < best_loss:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": total_loss,
                }
                best_loss = total_loss
                torch.save(checkpoint, name_model)
                logging.info(f"Saving model to {name_model} with loss = {best_loss}.")