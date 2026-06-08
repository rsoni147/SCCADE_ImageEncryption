import numpy as np
import cv2
from math import log10, sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import entropy
from skimage.metrics import structural_similarity


# =========================
# ENTROPY
# =========================
def calculate_entropy(image):
    """
    Calculate entropy of a single-channel image.
    """
    histogram = np.bincount(image.flatten(), minlength=256)

    probabilities = histogram / np.sum(histogram)

    # Remove zero probabilities to avoid log2(0)
    probabilities = probabilities[probabilities > 0]

    return entropy(probabilities, base=2)


# =========================
# NPCR
# =========================
def calculate_npcr_rgb(original_image, encrypted_image):

    npcr_values = {}

    for i, color in enumerate(['Blue', 'Green', 'Red']):

        different_pixels = (
            original_image[:, :, i] != encrypted_image[:, :, i]
        )

        npcr = (
            np.sum(different_pixels)
            / different_pixels.size
        ) * 100

        npcr_values[color] = npcr

    return npcr_values


# =========================
# SSIM
# =========================
def SSIM(altered_image, original_image):

    altered_gray = cv2.cvtColor(
        altered_image.astype(np.uint8),
        cv2.COLOR_BGR2GRAY
    )

    original_gray = cv2.cvtColor(
        original_image.astype(np.uint8),
        cv2.COLOR_BGR2GRAY
    )

    score = structural_similarity(
        original_gray,
        altered_gray,
        data_range=255
    )

    return score


# =========================
# MSE (Mean Squared Error)
# =========================
def calculate_mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1] * image1.shape[2])
    return err


# =========================
# PSNR (Peak Signal-to-Noise Ratio)
# =========================
def calculate_psnr(image1, image2):
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return 100 # Images are identical
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


# =========================
# UACI (Unified Averaging Changed Intensity)
# =========================
def calculate_uaci(original_image, encrypted_image):
    diff = np.sum(np.abs(original_image.astype("float") - encrypted_image.astype("float")))
    uaci = (diff / (original_image.size * 255)) * 100
    return uaci


# =========================
# Correlation Coefficient
# =========================
def calculate_correlation_coefficient(image1, image2):
    corr_values = {}
    for i, color in enumerate(['Blue', 'Green', 'Red']):
        flat1 = image1[:, :, i].flatten()
        flat2 = image2[:, :, i].flatten()
        corr_matrix = np.corrcoef(flat1, flat2)
        corr_values[color] = corr_matrix[0, 1]
    return corr_values

# =========================
# Vertical Correlation
# =========================
def calculate_vertical_correlation(image1, image2):
    corr_values = {}
    h, w, _ = image1.shape
    for c, color in enumerate(['Blue', 'Green', 'Red']):
        pixels1 = []
        pixels2 = []
        for i in range(h - 1):
            for j in range(w):
                pixels1.append(image1[i, j, c])
                pixels2.append(image1[i + 1, j, c])
        corr_matrix = np.corrcoef(pixels1, pixels2)
        corr_values[f'{color}_original'] = corr_matrix[0, 1]

        pixels1 = []
        pixels2 = []
        for i in range(h - 1):
            for j in range(w):
                pixels1.append(image2[i, j, c])
                pixels2.append(image2[i + 1, j, c])
        corr_matrix = np.corrcoef(pixels1, pixels2)
        corr_values[f'{color}_target'] = corr_matrix[0, 1]
    return corr_values

# =========================
# Horizontal Correlation
# =========================
def calculate_horizontal_correlation(image1, image2):
    corr_values = {}
    h, w, _ = image1.shape
    for c, color in enumerate(['Blue', 'Green', 'Red']):
        pixels1 = []
        pixels2 = []
        for i in range(h):
            for j in range(w - 1):
                pixels1.append(image1[i, j, c])
                pixels2.append(image1[i, j + 1, c])
        corr_matrix = np.corrcoef(pixels1, pixels2)
        corr_values[f'{color}_original'] = corr_matrix[0, 1]

        pixels1 = []
        pixels2 = []
        for i in range(h):
            for j in range(w - 1):
                pixels1.append(image2[i, j, c])
                pixels2.append(image2[i, j + 1, c])
        corr_matrix = np.corrcoef(pixels1, pixels2)
        corr_values[f'{color}_target'] = corr_matrix[0, 1]
    return corr_values

# =========================
# Diagonal Correlation
# =========================
def calculate_diagonal_correlation(image1, image2):
    corr_values = {}
    h, w, _ = image1.shape
    for c, color in enumerate(['Blue', 'Green', 'Red']):
        pixels1 = []
        pixels2 = []
        for i in range(h - 1):
            for j in range(w - 1):
                pixels1.append(image1[i, j, c])
                pixels2.append(image1[i + 1, j + 1, c])
        corr_matrix = np.corrcoef(pixels1, pixels2)
        corr_values[f'{color}_original'] = corr_matrix[0, 1]

        pixels1 = []
        pixels2 = []
        for i in range(h - 1):
            for j in range(w - 1):
                pixels1.append(image2[i, j, c])
                pixels2.append(image2[i + 1, j + 1, c])
        corr_matrix = np.corrcoef(pixels1, pixels2)
        corr_values[f'{color}_target'] = corr_matrix[0, 1]
    return corr_values

# =========================
# 3D RGB HISTOGRAM
# =========================
def plot_3d_rgb_histogram(image):
    """Generates a 3D histogram for the given image."""
    # Convert image to float32 for better precision
    image = np.float32(image)

    # Split channels
    b, g, r = cv2.split(image)

    # Flatten channels
    b = b.flatten()
    g = g.flatten()
    r = r.flatten()

    # Get bin centers
    r_edges, g_edges, b_edges = edges
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2
    g_centers = (g_edges[:-1] + g_edges[1:]) / 2
    b_centers = (b_edges[:-1] + b_edges[1:]) / 2

    # Get non-zero bins for visualization
    r_vals, g_vals, b_vals = np.meshgrid(r_centers, g_centers, b_centers, indexing='ij')
    r_vals = r_vals.flatten()
    g_vals = g_vals.flatten()
    b_vals = b_vals.flatten()
    hist_values = hist.flatten()

    # Filter out bins with no values
    nonzero_indices = hist_values > 0
    r_vals = r_vals[nonzero_indices]
    g_vals = g_vals[nonzero_indices]
    b_vals = b_vals[nonzero_indices]
    hist_values = hist_values[nonzero_indices]

    # Normalize histogram values for better visualization
    hist_values = hist_values / hist_values.max()

    # Create 3D scatter plot
    fig = plt.figure(figsize=(5, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot histogram points with color mapping
    ax.scatter(r_vals, g_vals, b_vals, c=np.column_stack((r_vals/255, g_vals/255, b_vals/255)), s=hist_values * 100)

    # Set labels and title
    ax.set_xlabel('Red Channel')
    ax.set_ylabel('Green Channel')
    ax.set_zlabel('Blue Channel')
    ax.set_title('3D RGB Histogram')

    plt.show()


# ====================================================
# EVALUATION
# ====================================================

# Load the image for histogram plotting (this part was duplicated, keeping only the calls)
# The 'image' variable is already defined from the main execution block (cell SUXzvfwpjnKL)
# The 'cipher' variable is also defined from the main execution block (cell SUXzvfwpjnKL)

plot_3d_rgb_histogram(image)
plot_3d_rgb_histogram(cipher)


# Original image entropy
entropy_image = {
    color: calculate_entropy(image[:, :, i])
    for i, color in enumerate(['Blue', 'Green', 'Red'])
}

print("\nOriginal Image Entropy:")
print(entropy_image)


# Cipher image entropy
entropy_cipher = {
    color: calculate_entropy(cipher[:, :, i])
    for i, color in enumerate(['Blue', 'Green', 'Red'])
}

print("\nCipher Image Entropy:")
print(entropy_cipher)


# NPCR
npcr_rgb = calculate_npcr_rgb(image, cipher)

print("\nNPCR for each channel:")
print(npcr_rgb)


# UACI
uaci_value = calculate_uaci(image, cipher)
print(f"\nUACI (Original vs Encrypted): {uaci_value:.4f}")


# Correlation Coefficient (Overall, Original vs Encrypted)
corr_coeff_cipher = calculate_correlation_coefficient(image, cipher)
print("\nCorrelation Coefficient (Overall, Original vs Encrypted) for each channel:")
print(corr_coeff_cipher)


# Vertical Correlation (Original vs Encrypted)
vert_corr_cipher = calculate_vertical_correlation(image, cipher)
print("\nVertical Correlation (Original vs Encrypted) for each channel:")
print(vert_corr_cipher)

# Horizontal Correlation (Original vs Encrypted)
horz_corr_cipher = calculate_horizontal_correlation(image, cipher)
print("\nHorizontal Correlation (Original vs Encrypted) for each channel:")
print(horz_corr_cipher)

# Diagonal Correlation (Original vs Encrypted)
diag_corr_cipher = calculate_diagonal_correlation(image, cipher)
print("\nDiagonal Correlation (Original vs Encrypted) for each channel:")
print(diag_corr_cipher)


# MSE (Original vs Encrypted)
mse_cipher = calculate_mse(image, cipher)
print(f"\nMSE (Original vs Encrypted): {mse_cipher:.4f}")


# PSNR (Original vs Encrypted)
psnr_cipher = calculate_psnr(image, cipher)
print(f"\nPSNR (Original vs Encrypted): {psnr_cipher:.4f} dB")


# SSIM for encrypted image
ssim_cipher = SSIM(cipher, image)

print(f"\nSSIM (Encrypted vs Original): {ssim_cipher:.4f}")


# Decrypted image entropy
entropy_restored = {
    color: calculate_entropy(unshuffled_image[:, :, i])
    for i, color in enumerate(['Blue', 'Green', 'Red'])
}

print("\nDecrypted Image Entropy:")
print(entropy_restored)


# SSIM for decrypted image
ssim_restored = SSIM(unshuffled_image, image)

print(f"\nSSIM (Decrypted vs Original): {ssim_restored:.4f}")


# Correlation Coefficient (Overall, Original vs Decrypted)
corr_coeff_restored = calculate_correlation_coefficient(image, unshuffled_image)
print("\nCorrelation Coefficient (Overall, Original vs Decrypted) for each channel:")
print(corr_coeff_restored)


# Vertical Correlation (Original vs Decrypted)
vert_corr_restored = calculate_vertical_correlation(image, unshuffled_image)
print("\nVertical Correlation (Original vs Decrypted) for each channel:")
print(vert_corr_restored)

# Horizontal Correlation (Original vs Decrypted)
horz_corr_restored = calculate_horizontal_correlation(image, unshuffled_image)
print("\nHorizontal Correlation (Original vs Decrypted) for each channel:")
print(horz_corr_restored)

# Diagonal Correlation (Original vs Decrypted)
diag_corr_restored = calculate_diagonal_correlation(image, unshuffled_image)
print("\nDiagonal Correlation (Original vs Decrypted) for each channel:")
print(diag_corr_restored)


# MSE (Original vs Decrypted)
mse_restored = calculate_mse(image, unshuffled_image)
print(f"\nMSE (Original vs Decrypted): {mse_restored:.4f}")


# PSNR (Original vs Decrypted)
psnr_restored = calculate_psnr(image, unshuffled_image)
print(f"\nPSNR (Original vs Decrypted): {psnr_restored:.4f} dB")
