import torch
from torch import nn
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from reporting import get_experiment_name
from torch import optim
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from dec.deep_embedded_clustering import DEC


def train_DEC_func_aligned(autoencoder,
                           dataloader,
                           dataloader_ind,
                           num_epochs,
                           criterion_rec,
                           criterion_cluster,
                           output_dir,
                           update_interval,
                           divergence_tolerance,
                           gamma,
                           learning_rate,
                           batch_size,
                           proximal,
                           num_clusters):
    """
    Training for deep embedded clustering.
    Step 1: Initialise cluster centres
    Step 2:
    :param num_clusters:
    :param autoencoder:
    :param dataloader:
    :param dataloader_ind:
    :param num_epochs:
    :param criterion_rec:
    :param criterion_cluster:
    :param output_dir:
    :param update_interval:
    :param divergence_tolerance:
    :param gamma:
    :param learning_rate:
    :param batch_size:
    :param proximal:
    :return:
    """
    if proximal == 0:
        prox = 'Distal'
    elif proximal == 1:
        prox = 'Proximal'
    else:
        prox = 'All'
    autoencoder.decoder_type = autoencoder.decoder_type + "DEC" + prox + f"clusters{num_clusters}_aligned"
    name_logging, name_model, name_writer, name = get_experiment_name(
        model=autoencoder, output_dir=output_dir
    )
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.basicConfig(filename=name_logging, level=logging.INFO)
    logging.info(f"Started training model {name} at {now}.")

    logging.info(f"Training on {prox} data with number of clusters set to {num_clusters}")
    writer = SummaryWriter(log_dir=name_writer)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)

    # Step 1: initialise cluster centres
    print("Initialising cluster centres and deciding the model")
    logging.info("Initialising cluster centres and returning the model")

    model = initialise_cluster_centres(autoencoder=autoencoder,
                                       dataloader_ind=dataloader_ind,
                                       device=device,
                                       num_clusters=num_clusters
                                       )
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate * 16 / batch_size,
        betas=(0.9, 0.999),
        weight_decay=1e-6,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    # Initialise q distribution and p distribution
    logging.info("Initialising q and p distributions")
    q_distribution, previous_predictions = calculate_q_distribution(model=model,
                                                                    dataloader_ind=dataloader_ind,
                                                                    device=device)
    p_distribution = calculate_p_distribution(q_distribution=q_distribution)

    model.to(device)
    model.train()
    best_loss = 1000000000
    niter = 1
    logging.info("Starting training")
    for epoch in range(num_epochs):
        batch_num = 1
        running_loss = 0.0
        if (epoch % update_interval == 0) and (epoch != 0):
            logging.info(f"Updating target distribution")
            q_distribution, predictions = calculate_q_distribution(model=model,
                                                                   dataloader_ind=dataloader_ind,
                                                                   device=device)
            p_distribution = calculate_p_distribution(q_distribution=q_distribution)
            delta_label, previous_predictions = check_tolerance(
                predictions, previous_predictions
            )
            logging.info(f"Delta label == {delta_label}")
            if delta_label < divergence_tolerance:
                print(
                    f"Label divergence {delta_label} < "
                    f"divergence tolerance {divergence_tolerance}"
                )
                print("Reached tolerance threshold. Stopping training.")
                logging.info(f"Label divergence {delta_label} < "
                             f"divergence tolerance {divergence_tolerance}"
                             f"Reached tolerance threshold. Stopping training.")
                break

        model.train()
        with tqdm(dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs = data[1]
                inputs = inputs.to(device)
                batch_size = inputs.shape[0]

                # ===================forward=====================
                with torch.set_grad_enabled(True):
                    output, features, q = model(inputs)
                    optimizer.zero_grad()
                    loss_rec = criterion_rec(output, inputs)
                    p = torch.from_numpy(
                        p_distribution[((batch_num - 1) * batch_size):(batch_num * batch_size), :]
                    ).to('cuda:0')
                    print(p.shape)
                    print(q.shape)
                    loss_cluster = criterion_cluster(torch.log(q), p)
                    loss = loss_rec + (gamma * loss_cluster)
                    # ===================backward====================
                    loss.backward()
                    optimizer.step()

                batch_loss = loss.detach().item() / batch_size
                batch_loss_rec = loss_rec.detach().item() / batch_size
                batch_loss_cluster = loss_cluster.detach().item() / batch_size
                running_loss += batch_loss
                batch_num += 1
                writer.add_scalar("/Loss", batch_loss, niter)
                niter += 1
                tepoch.set_postfix(loss=batch_loss,
                                   RecLoss=batch_loss_rec,
                                   ClusterLoss=batch_loss_cluster)

                if batch_num % 10 == 0:
                    logging.info(
                        f"[{epoch}/{num_epochs}]"
                        f"[{batch_num}/{len(dataloader)}]"
                        f"LossTot: {batch_loss}"
                        f"LossRec: {batch_loss_rec}"
                        f"LossCluster: {batch_loss_cluster}"
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

        # scheduler.step()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": total_loss,
    }
    torch.save(checkpoint, name_model)


def initialise_cluster_centres(autoencoder, dataloader_ind, device, num_clusters=None):
    """
    Initialise the cluster centres.
        Loop through the data to extract feature vectors for each data point.
        Perform elbow method to test the optimal number of clusters.
        Perform k-means with this optimal number and assign DEC model with this many clusters
        Add cluster centres as weights to new model
    :param num_clusters:
    :param autoencoder:
    :param dataloader_ind: dataloader with batch size = 1
    :param device:
    :return: DEC model with initialised cluster centres
    """
    features_all = []
    autoencoder.eval()

    with tqdm(dataloader_ind, unit="data") as tepoch:
        for data in tepoch:
            with torch.no_grad():
                inputs = data[1]
                inputs = inputs.to(device)
                output, features = autoencoder(inputs)
                features_all.append(torch.squeeze(features).cpu().detach().numpy())

    features_np = np.asarray(features_all)
    if num_clusters is None:
        wcss = []
        for i in range(1, 10):
            kmeans = KMeans(i)
            kmeans.fit(features_np)
            wcss_iter = kmeans.inertia_
            wcss.append(wcss_iter)

        kl = KneeLocator(
            range(1, 10), wcss, curve="convex", direction="decreasing"
        )
        number_clusters = kl.elbow
    else:
        number_clusters = num_clusters
    # logging.info(f"Optimal number of cluster: {number_clusters}")
    kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(features_np)
    weights = torch.from_numpy(kmeans.cluster_centers_)
    model = DEC(autoencoder=autoencoder, num_clusters=number_clusters)
    del autoencoder
    model.clustering_layer.set_weight(weights)

    return model


def calculate_q_distribution(model, dataloader_ind, device):
    """
    Calculates the q distribution of the embeddings
    :param model:
    :param dataloader_ind:
    :param device:
    :return:
    """
    q_distribution_all = []
    model.eval()
    with tqdm(dataloader_ind, unit="data") as tepoch:
        for data in tepoch:
            with torch.no_grad():
                inputs = data[1]
                inputs = inputs.to(device)
                output, features, q_distribution = model(inputs)
                q_distribution_all.append(torch.squeeze(q_distribution).cpu().detach().numpy())

    q_distribution_np = np.asarray(q_distribution_all)
    cluster_predictions = np.argmax(q_distribution_np.data, axis=1)

    return q_distribution_np, cluster_predictions


def calculate_p_distribution(q_distribution):
    """

    :param q_distribution:
    :return:
    """
    norm_squared_q = q_distribution ** 2 / np.sum(q_distribution, axis=0)
    p_distribution = np.transpose(np.transpose(norm_squared_q) / np.sum(norm_squared_q, axis=1))
    return p_distribution


def check_tolerance(cluster_predictions, previous_cluster_predictions):
    """

    :param cluster_predictions:
    :param previous_cluster_predictions:
    :return:
    """
    delta_label = (
            np.sum(cluster_predictions != previous_cluster_predictions).astype(
                np.float32
            )
            / cluster_predictions.shape[0]
    )
    previous_cluster_predictions = np.copy(cluster_predictions)
    return delta_label, previous_cluster_predictions
