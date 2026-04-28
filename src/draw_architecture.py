import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_simple_architecture():
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # Define steps in the architecture
    steps = [
        "Input MRI Image\n(224 x 224 x 3)",
        "Preprocessing &\nSkull Stripping",
        "DenseNet121 Base\n(Pre-trained on ImageNet)",
        "Fine-Tuned Layers\n(Unfrozen Conv Blocks)",
        "Global Average Pooling\n(GAP)",
        "Dropout Layer\n(Rate = 0.5)",
        "Dense Layer\n(512 Units, ReLU)",
        "Output Layer\n(4 Classes, Softmax)",
        "Prediction:\nGlioma, Meningioma, Pituitary, Normal"
    ]

    # Draw boxes
    box_width = 7
    box_height = 1.2
    start_x = 1.5
    start_y = 18.5
    y_step = 2

    for i, step in enumerate(steps):
        # Determine colors (like standard draw.io styles)
        if i == 0 or i == len(steps) - 1:
            facecolor = '#e1d5e7' # Light purple for input/output
            edgecolor = '#9673a6'
        elif i == 1:
            facecolor = '#fff2cc' # Light yellow for preprocessing
            edgecolor = '#d6b656'
        elif i == 2 or i == 3:
            facecolor = '#dae8fc' # Light blue for CNN base
            edgecolor = '#6c8ebf'
        else:
            facecolor = '#d5e8d4' # Light green for classifier head
            edgecolor = '#82b366'

        # Draw Rectangle
        rect = patches.Rectangle((start_x, start_y - i * y_step), box_width, box_height, 
                                 linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=2)
        ax.add_patch(rect)

        # Add text
        plt.text(start_x + box_width/2, start_y - i * y_step + box_height/2, step, 
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=11, fontfamily='sans-serif', fontweight='bold', color='#333333', zorder=3)

    # Draw arrows
    for i in range(len(steps) - 1):
        x_pos = start_x + box_width / 2
        y_top = start_y - i * y_step
        y_bottom = start_y - (i + 1) * y_step + box_height
        
        ax.annotate('', xy=(x_pos, y_bottom), xytext=(x_pos, y_top),
                    arrowprops=dict(arrowstyle="->", color='#666666', lw=2), zorder=1)

    plt.title("Simple Model Architecture (Flowchart)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('figures/simple_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    draw_simple_architecture()
    print("Simple architecture diagram generated successfully.")
