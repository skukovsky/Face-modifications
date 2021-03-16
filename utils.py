import matplotlib.pyplot as plt
from IPython.display import clear_output

def plot_gallery(input, output, n_samples):
    clear_output(wait=True)
    plt.figure(figsize=(15, 5))
    for idx in range(n_samples):
        plt.subplot(2, n_samples, idx+1)
        plt.imshow(np.clip(torch.reshape(input[idx], (45, 45, 3)), 0, 1))
        plt.title('Real')
        plt.axis('off')

        plt.subplot(2, n_samples, idx+(n_samples+1))
        plt.imshow(np.clip(torch.reshape(output[idx], (45, 45, 3)), 0, 1))
        plt.title('Output')
        plt.axis('off')
        # plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
    plt.show()
