TODO(example_dataset): Markdown description of your dataset.
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):

tfds.core.DatasetInfo(
    name='spot_carrot',
    full_name='spot_carrot/1.0.0',
    description="""
    TODO(example_dataset): Markdown description of your dataset.
    Description is **formatted** as markdown.
    
    It should also contain any processing which has been applied (if any),
    (e.g. corrupted example skipped, images cropped,...):
    """,
    homepage='https://www.tensorflow.org/datasets/catalog/spot_carrot',
    data_dir='/home/zhangy50/tensorflow_datasets/spot_carrot/1.0.0',
    file_format=tfrecord,
    download_size=Unknown size,
    dataset_size=45.87 MiB,
    features=FeaturesDict({
        'episode_metadata': FeaturesDict({
            'file_path': Text(shape=(), dtype=string),
        }),
        'steps': Dataset({
            'action': Tensor(shape=(8,), dtype=float32, description=Robot action, consists of [7x joint angles, 1x gripper on/off state].),
            'discount': Scalar(shape=(), dtype=float32, description=Discount if provided, default to 1.),
            'is_first': Scalar(shape=(), dtype=bool, description=True on first step of the episode.),
            'is_last': Scalar(shape=(), dtype=bool, description=True on last step of the episode.),
            'is_terminal': Scalar(shape=(), dtype=bool, description=True on last step of the episode if it is a terminal step, True for demos.),
            'language_embedding': Tensor(shape=(512,), dtype=float32, description=Kona language embedding. See https://tfhub.dev/google/universal-sentence-encoder-large/5),
            'language_instruction': Text(shape=(), dtype=string),
            'observation': FeaturesDict({
                'image': Image(shape=(224, 224, 3), dtype=uint8, description=Main camera RGB observation.),
                'state': Tensor(shape=(8,), dtype=float32, description=Robot state, consists of [7x robot joint angles, 1x gripper angle]),
                'wrist_image': Image(shape=(224, 224, 3), dtype=uint8, description=Wrist camera RGB observation.),
            }),
            'reward': Scalar(shape=(), dtype=float32, description=Reward if provided, 1 on final step for demos.),
        }),
    }),
    supervised_keys=None,
    disable_shuffling=False,
    nondeterministic_order=False,
    splits={
        'train': <SplitInfo num_examples=79, num_shards=1>,
        'val': <SplitInfo num_examples=21, num_shards=1>,
    },
    citation="""// TODO(example_dataset): BibTeX citation""",
)