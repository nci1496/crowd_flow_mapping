import sys
sys.path.insert(0, '.')

from main import SimulationRunner, create_test_config
from src.visualize import Visualizer
import matplotlib.pyplot as plt

# 配置
config = create_test_config()
runner = SimulationRunner(config)
runner.setup()

# 运行仿真
results = runner.run(num_steps=100, verbose=True)

# 保存图片
runner.plot_final_state(save_path='results/final_state.png')

# 保存动画 (使用 PillowWriter for GIF)
try:
    animation = runner.create_animation()
    runner.visualizer.save_animation(animation, 'results/simulation.gif', fps=10)
except Exception as e:
    print(f"动画保存跳过: {e}")

# 保存 JSON 数据
runner.save_results('results/simulation_results.json')

plt.close('all')
print("数据已保存到 results/ 目录")