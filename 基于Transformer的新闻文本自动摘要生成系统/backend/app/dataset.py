# -*- coding: utf-8 -*-
"""内置新闻示例数据集：3 条中文新闻 + 3 条英文新闻（含参考摘要）。

用于演示、评测与界面示例按钮。
"""
from typing import List, Dict

NEWS_SAMPLES: List[Dict[str, str]] = [
    {
        "id": 1,
        "title": "中国成功发射新一代通信卫星",
        "category": "科技",
        "text": "北京时间2024年3月15日，我国在文昌航天发射场成功发射了新一代通信技术试验卫星。该卫星采用东方红四号增强型平台，搭载了多项先进通信载荷，主要用于开展高速率、大容量通信技术验证。发射任务取得圆满成功，卫星顺利进入预定轨道。这是我国今年完成的第8次航天发射任务，进一步提升了我国在通信卫星领域的技术水平。专家表示，该卫星将为未来6G通信网络建设提供重要技术支撑。",
        "reference_summary": "我国在文昌成功发射新一代通信技术试验卫星，用于高速率大容量通信技术验证，将为6G网络建设提供技术支撑。",
    },
    {
        "id": 2,
        "title": "全国多地启动新一批保障性住房建设",
        "category": "社会",
        "text": "近日，北京、上海、广州、深圳等全国多个城市陆续启动了新一批保障性住房建设项目。这批项目主要针对新市民、青年人等住房困难群体，包括保障性租赁住房、共有产权住房等多种形式。各地政府表示，将严格落实土地、金融、财税等支持政策，确保项目按期保质完成。预计到2025年底，全国保障性住房覆盖率将显著提升，更多中低收入家庭的住房困难问题将得到有效缓解。",
        "reference_summary": "全国多地启动新一批保障性住房建设，面向新市民和青年人，涵盖租赁住房和共有产权住房，预计2025年底显著提升覆盖率。",
    },
    {
        "id": 3,
        "title": "央行宣布下调存款准备金率0.5个百分点",
        "category": "财经",
        "text": "中国人民银行今日宣布，决定于下周一下调金融机构存款准备金率0.5个百分点。此次降准为全面降准，预计释放长期资金约1.2万亿元。央行表示，此次降准旨在优化金融机构资金结构，提升金融服务实体经济能力，支持中小企业发展。市场分析认为，降准将有效降低银行负债成本，增强信贷投放能力，对稳定经济增长具有积极作用。这是央行今年第二次实施降准政策。",
        "reference_summary": "央行宣布全面降准0.5个百分点，预计释放长期资金约1.2万亿元，旨在提升金融服务实体经济能力，支持中小企业发展。",
    },
    {
        "id": 4,
        "title": "China's economy grows 5.2% in 2024",
        "category": "财经",
        "text": "China's gross domestic product grew 5.2 percent year on year in 2024, official data showed Wednesday, meeting the government's annual target of around 5 percent. The growth was driven by strong exports, recovering domestic consumption and a series of stimulus measures rolled out since September. Industrial output rose 5.8 percent while retail sales increased 3.5 percent. Economists said the data signals stable momentum, though challenges remain in the property sector and local government debt.",
        "reference_summary": "China's GDP grew 5.2% in 2024, meeting the annual target, driven by exports, consumption recovery and stimulus measures.",
    },
    {
        "id": 5,
        "title": "Federal Reserve cuts interest rates by 25 basis points",
        "category": "财经",
        "text": "The US Federal Reserve cut its benchmark interest rate by 25 basis points on Wednesday, bringing the target range to 4.25-4.5 percent. Fed Chair Jerome Powell said the decision reflects easing inflation and a cooling labor market, but signaled fewer cuts next year than previously expected. US stocks fell following the announcement as investors recalibrated expectations for monetary policy in 2025.",
        "reference_summary": "The Fed cut rates by 25 basis points to 4.25-4.5%, citing easing inflation, but signaled fewer cuts in 2025, causing stocks to fall.",
    },
    {
        "id": 6,
        "title": "Apple unveils new AI-powered iPhone",
        "category": "科技",
        "text": "Apple on Tuesday unveiled its latest iPhone lineup featuring a new chip designed to power artificial intelligence features, including an upgraded Siri assistant and advanced photo editing tools. The new models will go on sale next Friday, with prices starting at 799 US dollars. Analysts say the AI push is critical for Apple as it competes with rivals like Samsung and Google, which have already integrated generative AI into their flagship phones.",
        "reference_summary": "Apple unveiled new AI-powered iPhones with upgraded Siri and photo tools, starting at $799, as it competes in the generative AI race.",
    },
]


def get_all_samples() -> List[Dict[str, str]]:
    """返回全部新闻示例。"""
    return NEWS_SAMPLES


def get_sample_by_id(sample_id: int) -> Dict[str, str]:
    """按 ID 获取单条示例，不存在则返回 None。"""
    if 1 <= sample_id <= len(NEWS_SAMPLES):
        return NEWS_SAMPLES[sample_id - 1]
    return None
