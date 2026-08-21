import numpy as np

def alignment_of_coral_technique(source_of_x: np.ndarray, x_target: np.ndarray, eps: float = 1e-5, alpha: float = 1.0) -> np.ndarray:
    if alpha <= 0:
        return source_of_x.copy()
    features_of_n = source_of_x.shape[1]
    covariance_source = np.cov(source_of_x, rowvar=False) + eps * np.eye(features_of_n)
    covariance_target = np.cov(x_target, rowvar=False) + eps * np.eye(features_of_n)

    def square_root_of_matrix(matrix):
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.clip(eigenvalues, a_min=eps, a_max=None)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

    def inverse_square_root_of_matrix(matrix):
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.clip(eigenvalues, a_min=eps, a_max=None)
        return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    covariance_source_inverse_square_root = inverse_square_root_of_matrix(covariance_source)
    covariance_target_square_root = square_root_of_matrix(covariance_target)
    whitened_source_of_x = source_of_x @ covariance_source_inverse_square_root
    fully_aligned_source_of_x = whitened_source_of_x @ covariance_target_square_root

    if alpha >= 1.0:
        return fully_aligned_source_of_x
    return (1 - alpha) * source_of_x + alpha * fully_aligned_source_of_x