#  Copyright 2019 Gabriele Valvano
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import tensorflow as tf
tf.random.set_seed(12345)
import numpy as np
np.random.seed(12345)
import random
random.seed(12345)
from idas.logger import telegram_notifier
import logging
import time
import config
import importlib


# telegram bot ---
TELEGRAM_TOKEN = '696365945:AAEZgDVuEkc7SF1iqbT0zR2YolbCvUwdfT4'  # token-id
TELEGRAM_CHAT_ID = '171620634'  # chat-id
# ----------------


def parse_model_type(args):
    """ Import the correct model for the experiments """
    experiment_type = args.experiment_type
    experiment = args.experiment
    dataset_name = args.dataset_name

    allowed_types = ['semi']
    assert experiment_type in allowed_types

    if experiment_type == 'semi':
        model = importlib.import_module('experiments_semi.{0}.{1}'.format(dataset_name, experiment)).Experiment()

    else:
        raise ValueError('Unsupported experiment type. Experiment type must be on in {0}.'.format(allowed_types))

    return model


def main():

    args = config.define_flags()

    # import the correct model for the experiment
    model = parse_model_type(args=args)
    model.build()
    writer = tf.summary.create_file_writer(model.log_dir)

    # load best model and do a test:
    model.load_pre_trained_model(model.checkpoint_dir, verbose=True)

    start_time = time.time()

    if args.notify:
        try:
            model.test(writer, paired_data=model.test_paired_data)
            tel_message = 'Training finished.'
        except Exception as exc:
            print(exc)
            tel_message = str(exc) + '\nAn error arised. Check your code!'

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Telegram notification:
        msg = "Automatic message from server 'imt_lucca'\n{separator}\n" \
              "<b>RUN_ID: </b>\n<pre> {run_id} </pre>\n" \
              "<b>MESSAGE: </b>\n<pre> {message} </pre>".format(run_id=model.run_id,
                                                                message=tel_message,
                                                                separator=' -' * 10)
        telegram_notifier.basic_notifier(logger_name='training_notifier',
                                         token_id=TELEGRAM_TOKEN,
                                         chat_id=TELEGRAM_CHAT_ID,
                                         message=msg,
                                         level=logging.INFO)
    else:
        model.test(writer, paired_data=model.test_paired_data)

    delta_t = time.time() - start_time
    print('\nTook: {0:.3f} hours'.format(delta_t/3600))


# parses flags and calls the `main` function above
if __name__ == '__main__':
    # tf.compat.v1.app.run()
    main()
