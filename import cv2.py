
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_histogram(channel):
    return cv2.calcHist([channel], [0], None, [256], [0, 256])

def plot_histogram(ax, hist, color, title):
    ax.plot(hist, color=color)
    ax.set_title(title)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Frequency")

def adjust_contrast(image, alpha=2.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def adjust_brightness(image, alpha=1, beta=-100):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def split_and_compute_histograms(image):
    b, g, r = cv2.split(image)
    hist_b = compute_histogram(b)
    hist_g = compute_histogram(g)
    hist_r = compute_histogram(r)
    return b, g, r, hist_b, hist_g, hist_r

# Load image
image_path ="D:\course\WhatsApp Image 2024-05-04 at 23.28.09_ed6e2636.jpg"
image = cv2.imread(image_path)

# Splitting channels and computing histograms
b, g, r, hist_b_before, hist_g_before, hist_r_before = split_and_compute_histograms(image)

# Plotting original image and its histograms
fig, axs = plt.subplots(3, 4, figsize=(15, 10))

# Original Image
axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title("Original Image")
axs[0, 0].axis("off")

plot_histogram(axs[0, 1], hist_b_before, "blue", "Blue Channel Histogram (Before)")
plot_histogram(axs[0, 2], hist_g_before, "green", "Green Channel Histogram (Before)")
plot_histogram(axs[0, 3], hist_r_before, "red", "Red Channel Histogram (Before)")

# Adjust contrast
contrast_image = adjust_contrast(image)

# Splitting channels and computing histograms after contrast adjustment
b_contrast, g_contrast, r_contrast, hist_b_after_contrast, hist_g_after_contrast, hist_r_after_contrast = split_and_compute_histograms(contrast_image)

# Plotting image after contrast adjustment and its histograms
axs[1, 0].imshow(cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title("Image after Increasing Contrast")
axs[1, 0].axis("off")

plot_histogram(axs[1, 1], hist_b_after_contrast, "blue", "Blue Channel Histogram (After)")
plot_histogram(axs[1, 2], hist_g_after_contrast, "green", "Green Channel Histogram (After)")
plot_histogram(axs[1, 3], hist_r_after_contrast, "red", "Red Channel Histogram (After)")

# Adjust brightness
brightness_image = adjust_brightness(contrast_image)

# Splitting channels and computing histograms after brightness adjustment
b_brightness, g_brightness, r_brightness, hist_b_after_brightness, hist_g_after_brightness, hist_r_after_brightness = split_and_compute_histograms(brightness_image)

# Plotting image after brightness adjustment and its histograms
axs[2, 0].imshow(cv2.cvtColor(brightness_image, cv2.COLOR_BGR2RGB))
axs[2, 0].set_title("Image after Decreasing Brightness")
axs[2, 0].axis("off")

plot_histogram(axs[2, 1], hist_b_after_brightness, "blue", "Blue Channel Histogram (After)")
plot_histogram(axs[2, 2], hist_g_after_brightness, "green", "Green Channel Histogram (After)")
plot_histogram(axs[2, 3], hist_r_after_brightness, "red", "Red Channel Histogram (After)")

plt.tight_layout()
plt.show()