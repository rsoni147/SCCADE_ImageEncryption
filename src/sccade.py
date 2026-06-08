import cv2
import numpy as np
from typing import Tuple
from google.colab.patches import cv2_imshow
from google.colab import drive


def generate_key(henon_params: Tuple[float], lorenz_params: Tuple[float], iter=5):
    a, b = 1.4, 0.3
    x_henon, y_henon = henon_params

    sigma, rho, beta = 10, 28, 8/3
    x_lorenz, y_lorenz, z_lorenz = lorenz_params

    for _ in range(iter):
        x_henon_next = 1 - a * x_henon**2 + y_henon
        y_henon_next = b * x_henon
        x_henon, y_henon = x_henon_next, y_henon_next

        dt = 0.01
        dx = sigma * (y_lorenz - x_lorenz) * dt
        dy = (x_lorenz * (rho - z_lorenz) - y_lorenz) * dt
        dz = (x_lorenz * y_lorenz - beta * z_lorenz) * dt
        x_lorenz += dx
        y_lorenz += dy
        z_lorenz += dz

    x_henon_int = int((x_henon % 1) * 2**64)
    x_lorenz_int = int((x_lorenz % 1) * 2**64)

    combined_value_dna = (x_henon_int << 64) | x_lorenz_int
    combined_value_soliton = (x_lorenz_int << 64) | x_henon_int

    dna_128bit = combined_value_dna & ((1 << 128) - 1)
    soliton_128bit = combined_value_soliton & ((1 << 128) - 1)

    return dna_128bit, soliton_128bit


def dna_diffusion(seed):
    nucleotides = ['A', 'T', 'C', 'G']
    binary_seed = bin(seed)[2:].zfill(128)
    half_length = len(binary_seed) // 2
    first_half = binary_seed[:half_length]
    second_half = binary_seed[half_length:]

    paired_sequence = ''.join([nucleotides[int(first_half[i:i+2], 2)] + nucleotides[int(second_half[i:i+2], 2)]
                               for i in range(0, half_length, 2)])

    def dna_to_binary(dna_sequence):
        #binary_mapping = {'T': '00', 'A': '01', 'G': '10', 'C': '11'}
        binary_mapping = {'T': '11', 'A': '00', 'G': '10', 'C': '01'}
        return ''.join(binary_mapping[nucleotide] for nucleotide in dna_sequence)

    diffused_binary = dna_to_binary(paired_sequence)
    diffused_seed = int(diffused_binary[:128], 2)
    return diffused_seed

def soliton_diffusion(seed, soliton_wave):
    binary_seed = bin(seed)[2:].zfill(128)
    #soliton_influence = np.floor(soliton_wave * 2).astype(int) % 2
    soliton_influence = np.floor(soliton_wave * 1e6).astype(int) % 2
    diffused_seed = int(''.join(str(int(binary_seed[i]) ^ soliton_influence[i % len(soliton_influence)])
                                for i in range(len(binary_seed))), 2)
    return diffused_seed

def generate_pseudo_random_sequence(key, num_blocks):
    key_upper = (key >> 64) & 0xFFFFFFFFFFFFFFFF
    key_lower = key & 0xFFFFFFFFFFFFFFFF
    random_sequence = [(key_upper * i + key_lower) & 0xFFFFFFFFFFFFFFFF for i in range(num_blocks)]
    shuffle_order = np.argsort(random_sequence)
    return shuffle_order

def block_shuffle(image, key, block_size):
    h, w = image.shape[:2]
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    num_blocks = num_blocks_h * num_blocks_w
    shuffle_order = generate_pseudo_random_sequence(key, num_blocks)
    shuffled_image = np.zeros_like(image)

    for idx, new_idx in enumerate(shuffle_order):
        y_old = (idx // num_blocks_w) * block_size
        x_old = (idx % num_blocks_w) * block_size
        y_new = (new_idx // num_blocks_w) * block_size
        x_new = (new_idx % num_blocks_w) * block_size
        shuffled_image[y_new:y_new + block_size, x_new:x_new + block_size] = image[y_old:y_old + block_size, x_old:x_old + block_size]

    return shuffled_image

def block_unshuffle(shuffled_image, key, block_size):
    h, w = shuffled_image.shape[:2]
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    num_blocks = num_blocks_h * num_blocks_w
    shuffle_order = generate_pseudo_random_sequence(key, num_blocks)
    unshuffled_image = np.zeros_like(shuffled_image)

    for idx, new_idx in enumerate(shuffle_order):
        y_old = (idx // num_blocks_w) * block_size
        x_old = (idx % num_blocks_w) * block_size
        y_new = (new_idx // num_blocks_w) * block_size
        x_new = (new_idx % num_blocks_w) * block_size
        unshuffled_image[y_old:y_old + block_size, x_old:x_old + block_size] = shuffled_image[y_new:y_new + block_size, x_new:x_new + block_size]

    return unshuffled_image

def apply_rule(bin_num: list[int], leftmost=0) -> list[int]:
    new_bin_num = [0]*8
    new_bin_num[0] = bin_num[0] ^ leftmost
    new_bin_num[1] = int(not bin_num[1])
    new_bin_num[2] = int(not bin_num[2])
    new_bin_num[3] = int(not bin_num[3])
    new_bin_num[4] = int(not (bin_num[3]^bin_num[4]))
    new_bin_num[5] = bin_num[4] ^ bin_num[5] ^ bin_num[6]
    new_bin_num[6] = bin_num[5] ^ bin_num[6] ^ bin_num[7]
    new_bin_num[7] = int(not (bin_num[6]^bin_num[7]))
    return new_bin_num

def applyCA(image: np.ndarray, num_iters, encrypt=True):
    flat = image.flatten()
    dims = len(flat)
    left = 0
    for i in range(dims):
        num_bin = list(map(int, list(f'{flat[i]:08b}')))
        new_bin = apply_rule(num_bin, left)
        for _ in range(num_iters-1):
            new_bin = apply_rule(new_bin, left)
        flat[i] = int("".join(map(str, new_bin)), 2)
        left = num_bin[-1] if encrypt else new_bin[-1]
    return flat.reshape(image.shape)

def encrypt(image, dna_key, soliton_key):
    shuffled_image = block_shuffle(image, soliton_key, block_size=1)
    transformed_image = applyCA(shuffled_image, 2)
    cipher = encrypt_xor(transformed_image, dna_key)
    return cipher

def decrypt(cipher, dna_key, soliton_key):
    rev_image = decrypt_xor(cipher, dna_key)
    retransformed = applyCA(rev_image, 2, False)
    unshuffled_image = block_unshuffle(retransformed, soliton_key, block_size=1)
    return unshuffled_image

# ---------------------------------------------------------------
# MAIN EXECUTION BLOCK FOR COLAB
# ---------------------------------------------------------------

# --- Manual Encrypted Image Settings ---
use_manual_encrypted_image = False # Set to True if you want to provide your own encrypted image
manual_encrypted_image_path = "/content/drive/MyDrive/land10.png" # Specify the path to your manual encrypted image
manual_original_image_path = "/content/drive/MyDrive/my_image.png" # Specify the path to the ORIGINAL unencrypted image corresponding to the encrypted one
# ---------------------------------------

# Initialize image and cipher variables
image = None
cipher = None

if use_manual_encrypted_image:
    print(f"Loading original image from: {manual_original_image_path}")
    image = cv2.imread(manual_original_image_path)
    if image is None:
        print(f"Error: Could not load manual original image from {manual_original_image_path}. Please check the path and file.")
        # Consider raising an error or exiting here if loading fails
    print(f"Loading manual encrypted image from: {manual_encrypted_image_path}")
    cipher = cv2.imread(manual_encrypted_image_path)
    if cipher is None:
        print(f"Error: Could not load manual encrypted image from {manual_encrypted_image_path}. Please check the path and file.")
        # Consider raising an error or exiting here if loading fails
else:
    choice_image = '7'
    image_paths = {
        '1': "/content/drive/MyDrive/Colab Notebooks/Baboon.jpeg",
        '2': "/content/drive/MyDrive/Colab Notebooks/Lena.jpg",
        '3': "/content/drive/MyDrive/Colab Notebooks/Peppers.jpeg",
        '4': "/content/drive/MyDrive/Colab Notebooks/Boat.jpg",
        '5': "/content/drive/MyDrive/Colab Notebooks/Penguin.jpeg",
        '6': "/content/drive/MyDrive/Colab Notebooks/Bridge.png",
        '7': "/content/drive/MyDrive/landscape_image.png"
    }
    image_path = image_paths[choice_image]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}. Please check the path and file.")
        # Consider raising an error or exiting here if loading fails

# Ensure image_copy_1 and image_copy_2 are initialized regardless of path
if image is not None:
    image_copy_1 = image.copy()
    image_copy_2 = image.copy()
else: # If image loading failed, these would be problematic. Add a safety
    image_copy_1 = np.zeros((1,1,3), dtype=np.uint8) # Dummy image to prevent error
    image_copy_2 = np.zeros((1,1,3), dtype=np.uint8) # Dummy image to prevent error


import os


KEY_FILE = "/content/drive/MyDrive/encryption_keys.npz"

henon_params = None
lorenz_params = None

if use_manual_encrypted_image:
    # When using a manual encrypted image, we assume the keys for its encryption
    # are either already in KEY_FILE, or we'll generate random ones (which is usually wrong).
    # For now, stick to loading if available, otherwise generate.
    if os.path.exists(KEY_FILE):
        data = np.load(KEY_FILE)
        henon_params = data['henon_params']
        lorenz_params = data['lorenz_params']
        print("Loaded saved keys for manual decryption attempt.")
    else:
        # This path is problematic for user-provided encrypted images.
        # It generates *new* random keys, which won't match the user's encrypted image.
        henon_params = np.random.rand(2)
        lorenz_params = np.random.rand(3)
        np.savez(KEY_FILE, henon_params=henon_params, lorenz_params=lorenz_params)
        print("Generated and saved NEW random keys for manual decryption (WARNING: these might not match the original encryption keys of your manually provided image).")
else:
    # If not using a manual encrypted image, we are performing a fresh encryption/decryption.
    # In this scenario, we generate NEW keys and save them.
    henon_params = np.random.rand(2)
    lorenz_params = np.random.rand(3)
    np.savez(KEY_FILE, henon_params=henon_params, lorenz_params=lorenz_params)
    print("Generated and saved NEW keys for internal encryption/decryption.")



pr_128bit_dna, pr_128bit_soliton = generate_key(henon_params, lorenz_params)
dna_key = dna_diffusion(pr_128bit_dna)

c = 1.0
x = np.linspace(0, 7, 25)
t = 5
soliton_wave = 0.5 * c * (1 / np.cosh(0.5 * np.sqrt(c) * (x - c * t)))**2
soliton_key = soliton_diffusion(pr_128bit_soliton, soliton_wave)

print("DNA Diffused Key (128-bit):", dna_key)
print("Soliton Diffused Key (128-bit):", soliton_key)

# If not using manual_encrypted_image, cipher needs to be generated here
if not use_manual_encrypted_image:
    cipher = encrypt(image_copy_1, dna_key, soliton_key)

cipher_copy = cipher.copy()

unshuffled_image = decrypt(cipher_copy, dna_key, soliton_key)

# Show original, encrypted, and decrypted images
print("Original Image:")
if image is not None:
    cv2_imshow(image)
else:
    print("Original image could not be loaded.")

print("Encrypted Image:")
if cipher is not None:
    cv2_imshow(cipher)
else:
    print("Encrypted image could not be loaded.")

print("Decrypted Image:")
if unshuffled_image is not None:
    cv2_imshow(unshuffled_image)
else:
    print("Decrypted image could not be generated.")
