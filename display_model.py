import yaml, torch
from Utils.io_utils import instantiate_from_config
from ema_pytorch import EMA

# 1) 构建模型（与训练时配置一致）
with open('./Config/jumping.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
model = instantiate_from_config(cfg['model'])

# 2) 加载 checkpoint（主模型 & EMA）
ckpt = torch.load('./Result/Checkpoints_jumping_500/checkpoint-1.pt', map_location='cpu')
model.load_state_dict(ckpt['model'])

# 3) 恢复 EMA 对象并拿到平均权重的“影子模型”
ema = EMA(model, beta=cfg['solver']['ema']['decay'], update_every=cfg['solver']['ema']['update_interval'])
ema.load_state_dict(ckpt['ema'])
avg_model = ema.ema_model  # 这里就是 EMA 平均权重的模型

# 4) 读取 DecoderBlock / EncoderBlock 的参数（以前两层为例）
dec_blocks = avg_model.model.decoder.blocks
enc_blocks = avg_model.model.encoder.blocks

for i in range(min(2, len(dec_blocks))):
    blk = dec_blocks[i]
    # 解码器可能未启用频域注意力（老模型或配置未开启），因此需做属性判断
    if hasattr(blk, 'gate_f'):
        print(f'[DecoderBlock {i}] trend_scale={blk.trend_scale.item():.6f}, '
              f'season_scale={blk.season_scale.item():.6f}, '
              f'gate_f={blk.gate_f.item():.6f}')
    else:
        print(f'[DecoderBlock {i}] trend_scale={blk.trend_scale.item():.6f}, '
              f'season_scale={blk.season_scale.item():.6f}, '
              f'gate_f=NA (freq-attn disabled)')

for i in range(len(enc_blocks)):
    blk = enc_blocks[i]
    if hasattr(blk, 'gate_f'):
        print(f'[EncoderBlock {i}] gate_f={blk.gate_f.item():.6f}')