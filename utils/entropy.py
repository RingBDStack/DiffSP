
import torch
import numpy as np

epsilon = 1e-10

def min_max_normalize(x):
    min_val = x.min(dim=1, keepdim=True)[0]
    max_val = x.max(dim=1, keepdim=True)[0]
    range_val = max_val - min_val
    range_val = torch.clamp(range_val, min=1e-8)

    normalized_x = (x - min_val) / range_val
    return normalized_x

def rbf_kernel(x, sigma=0.5):
    # sq_dists = torch.sum(X**2, axis=1).reshape(-1, 1) + torch.sum(X**2, axis=1) - 2 * torch.mm(X, X.T)
    # K = torch.exp(-sq_dists / (2 * sigma**2))
    x = min_max_normalize(x)
    sq_dist = torch.cdist(x, x, p=2) ** 2
    K = torch.exp(-sq_dist / (2 * sigma ** 2))
    if torch.isnan(K).any() or torch.isinf(K).any():
        print(sq_dist)
        print("======")
        exit()
    return K

def normalize_npd(K):
    # n = K.shape[0]
    # A = torch.zeros_like(K)
    # for i in range(n):
    #     for j in range(n):
    #         A[i, j] = K[i, j] / (torch.sqrt(K[i, i] * K[j, j]) + epsilon)
    # A = A / n
    # print(K, K.shape)
    # print('---------------------------')
    n = K.shape[0]
    K_diag = torch.diag(K)
    K_diag_sqrt = torch.sqrt(K_diag.unsqueeze(1) * K_diag.unsqueeze(0))  # 计算 sqrt(K_{ii} * K_{jj})
    A = (1 / n) * K / K_diag_sqrt  # 计算 A_{ij}
    return A

def renyi_alpha_entropy(K, alpha=2):
    """
    Rényi's α-entropy: Eq(2) in "Multivariate Extension of Matrix-based R ́ enyi’s α-order Entropy Functional"
    K: (n, n)
    """
    A = normalize_npd(K)
    
    # eigenvalues
    eigenvalues = torch.linalg.eigvalsh(A)
    
    # Rényi's α-entropy
    sum_eigenvalues_alpha = torch.sum(eigenvalues ** alpha)
    S_alpha = (1 / (1 - alpha)) * torch.log2(sum_eigenvalues_alpha)
    
    return S_alpha


def joint_entropy(Grams):
    """
    Grams: the list of Gram Matrix Eq.(6)
    """
    A = Grams[0]
    # back_A = A.clone()
    for K in Grams[1:]:
        A = A * K
    A = A / torch.trace(A)
    try:
        j_entropy = renyi_alpha_entropy(A)
    except:
        print("error=============")
        exit()
    if torch.isinf(torch.tensor(j_entropy).clone().detach()):
        print("error=============")
        exit()
    return j_entropy


def transfer_entropy(A, A_t, A_t_1, kernel=True, normalize=True):
    # print(A, A_t, A_t_1); exit()
    if kernel:
        A, A_t, A_t_1 = rbf_kernel(A), rbf_kernel(A_t), rbf_kernel(A_t_1)
    if normalize:
        A, A_t, A_t_1 = normalize_npd(A), normalize_npd(A_t), normalize_npd(A_t_1)
    H_a_1 = joint_entropy([A_t, A_t_1])
    H_a_2 = renyi_alpha_entropy(A_t_1)
    H_a_3 = joint_entropy([A_t, A_t_1, A])
    H_a_4 = joint_entropy([A_t_1, A])
    te_a = H_a_1 - H_a_2 - H_a_3 + H_a_4
    return te_a

