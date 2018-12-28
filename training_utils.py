from DataGeneratorNew import DataGeneratorNew

import random

def train(train_data_xy, validation_data_xy, model, batch_size, G):
    training_generator = DataGeneratorNew(list(map(lambda ep: ep[0], train_data_xy)), 
                                        list(map(lambda ep: ep[1], train_data_xy)), 
                                        batch_size=batch_size, shuffle=True, model=model, graph=G)

    validation_generator = DataGeneratorNew(list(map(lambda ep: ep[0], validation_data_xy)), 
                                        list(map(lambda ep: ep[1], validation_data_xy)), 
                                        batch_size=batch_size, shuffle=True, model=model, graph=G)

    model.fit_generator(generator=training_generator,
            validation_data=validation_generator,
            use_multiprocessing=True,
            workers=10, epochs=15)

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def bucketing(bucket_size, episodes, placeholder_timestep):
    episode_batches = list(chunks(episodes, bucket_size))

    # Shuffle batches to fight overfitting (as they're passed here ordered by length)
    random.shuffle(episode_batches)

    # Pad each chunk
    for episode_batch in episode_batches:
        max_t = len(max(map(lambda e: e[1], episode_batch), key=len))

        for episode in episode_batch:
            to_append = max_t - len(episode[1])

            for _ in range(to_append):
                episode[0].append(placeholder_timestep[0])
                episode[1].append(placeholder_timestep[1])

    return episodes