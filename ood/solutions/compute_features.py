def compute_features(data_loader, model, device):
    features = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            features.extend(model(images, return_features=True).cpu().numpy())
    return np.array(features)


fit_features = compute_features(train_loader)
test_features_negatives = compute_features(cifar_test_loader)
test_features_positives = compute_features(svhn_test_loader)