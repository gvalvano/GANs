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

import numpy as np
np.random.seed(1234)
import random
random.seed(1234)


def get_splits():
    """""
    Returns an array of splits into validation, test and train indices.
    """
    # l = ['strasbourg', 'weimar', 'aachen', 'tubingen', 'jena', 'bochum', 'darmstadt', 'dusseldorf', 'hamburg',
    #   'cologne', 'monchengladbach', 'ulm', 'hanover', 'stuttgart', 'erfurt', 'krefeld', 'zurich', 'bremen']

    splits = {

        # ------------------------------------------------------------------------------------------------------------
        # splits with just 1 city in the labeled dataset

        '1vols': {
            'split0': {'test': ['munster', 'lindau', 'frankfurt'],
                       'validation': ['krefeld', 'zurich', 'bremen'],
                       'train_unsup': ['strasbourg', 'weimar', 'aachen', 'tubingen', 'jena', 'bochum',
                                       'darmstadt', 'dusseldorf'],
                       'train_disc': ['hamburg', 'cologne', 'monchengladbach', 'ulm', 'hanover', 'stuttgart', 'erfurt'],
                       'train_sup': ['hamburg']
                       },
            'split1': {'test': ['munster', 'lindau', 'frankfurt'],
                       'validation': ['stuttgart', 'zurich', 'weimar'],
                       'train_unsup': ['cologne', 'hamburg', 'aachen', 'monchengladbach', 'ulm', 'bochum',
                                       'darmstadt', 'erfurt'],
                       'train_disc': ['krefeld', 'bremen', 'strasbourg', 'tubingen', 'jena', 'hanover', 'dusseldorf'],
                       'train_sup': ['bochum']
                       },
            'split2': {'test': ['munster', 'lindau', 'frankfurt'],
                       'validation': ['strasbourg', 'weimar', 'aachen'],
                       'train_unsup': ['hamburg', 'cologne', 'monchengladbach', 'tubingen', 'jena', 'darmstadt',
                                       'erfurt', 'dusseldorf'],
                       'train_disc': ['krefeld', 'zurich', 'bremen', 'ulm', 'hanover', 'stuttgart', 'bochum'],
                       'train_sup': ['cologne']
                       },
        },

        # ------------------------------------------------------------------------------------------------------------
        # splits with just 3 cities in the labeled dataset

        '3vols': {
            'split0': {'test': ['munster', 'lindau', 'frankfurt'],
                       'validation': ['krefeld', 'zurich', 'bremen'],
                       'train_unsup': ['strasbourg', 'weimar', 'aachen', 'tubingen', 'jena', 'bochum',
                                       'darmstadt', 'dusseldorf'],
                       'train_disc': ['hamburg', 'cologne', 'monchengladbach', 'ulm', 'hanover', 'stuttgart', 'erfurt'],
                       'train_sup': ['hamburg', 'aachen', 'strasbourg']
                       },
            'split1': {'test': ['munster', 'lindau', 'frankfurt'],
                       'validation': ['stuttgart', 'zurich', 'weimar'],
                       'train_unsup': ['cologne', 'hamburg', 'aachen', 'monchengladbach', 'ulm', 'bochum',
                                       'darmstadt', 'erfurt'],
                       'train_disc': ['krefeld', 'bremen', 'strasbourg', 'tubingen', 'jena', 'hanover', 'dusseldorf'],
                       'train_sup': ['bochum', 'ulm', 'darmstadt']
                       },
            'split2': {'test': ['munster', 'lindau', 'frankfurt'],
                       'validation': ['strasbourg', 'weimar', 'aachen'],
                       'train_unsup': ['hamburg', 'cologne', 'monchengladbach', 'tubingen', 'jena', 'darmstadt',
                                       'erfurt', 'dusseldorf'],
                       'train_disc': ['krefeld', 'zurich', 'bremen', 'ulm', 'hanover', 'stuttgart', 'bochum'],
                       'train_sup': ['cologne', 'erfurt', 'monchengladbach']
                       },
        },

        # ------------------------------------------------------------------------------------------------------------
        # splits with just 5 cities in the labeled dataset

        '5vols': {
            'split0': {'test': ['munster', 'lindau', 'frankfurt'],
                       'validation': ['krefeld', 'zurich', 'bremen'],
                       'train_unsup': ['strasbourg', 'weimar', 'aachen', 'tubingen', 'jena', 'bochum',
                                       'darmstadt', 'dusseldorf'],
                       'train_disc': ['hamburg', 'cologne', 'monchengladbach', 'ulm', 'hanover', 'stuttgart', 'erfurt'],
                       'train_sup': ['hamburg', 'aachen', 'strasbourg', 'weimar', 'jena']
                       },
            'split1': {'test': ['munster', 'lindau', 'frankfurt'],
                       'validation': ['stuttgart', 'zurich', 'weimar'],
                       'train_unsup': ['cologne', 'hamburg', 'aachen', 'monchengladbach', 'ulm', 'bochum',
                                       'darmstadt', 'erfurt'],
                       'train_disc': ['krefeld', 'bremen', 'strasbourg', 'tubingen', 'jena', 'hanover', 'dusseldorf'],
                       'train_sup': ['bochum', 'ulm', 'darmstadt', 'erfurt', 'aachen']
                       },
            'split2': {'test': ['munster', 'lindau', 'frankfurt'],
                       'validation': ['strasbourg', 'weimar', 'aachen'],
                       'train_unsup': ['hamburg', 'cologne', 'monchengladbach', 'tubingen', 'jena', 'darmstadt',
                                       'erfurt', 'dusseldorf'],
                       'train_disc': ['krefeld', 'zurich', 'bremen', 'ulm', 'hanover', 'stuttgart', 'bochum'],
                       'train_sup': ['cologne', 'erfurt', 'monchengladbach', 'hamburg', 'tubingen']
                       },
        },

        # ------------------------------------------------------------------------------------------------------------
        # splits with just 6 cities in the labeled dataset

        '6vols': {
            'split0': {'test': ['munster', 'lindau', 'frankfurt'],
                       'validation': ['krefeld', 'zurich', 'bremen'],
                       'train_unsup': ['strasbourg', 'weimar', 'aachen', 'tubingen', 'jena', 'bochum',
                                       'darmstadt', 'dusseldorf'],
                       'train_disc': ['hamburg', 'cologne', 'monchengladbach', 'ulm', 'hanover', 'stuttgart', 'erfurt'],
                       'train_sup': ['hamburg', 'erfurt', 'aachen', 'bochum', 'strasbourg', 'weimar', 'jena']
                       },
            'split1': {'test': ['munster', 'lindau', 'frankfurt'],
                       'validation': ['stuttgart', 'zurich', 'weimar'],
                       'train_unsup': ['cologne', 'hamburg', 'aachen', 'monchengladbach', 'ulm', 'bochum',
                                       'darmstadt', 'erfurt'],
                       'train_disc': ['krefeld', 'bremen', 'strasbourg', 'tubingen', 'jena', 'hanover', 'dusseldorf'],
                       'train_sup': ['bochum', 'ulm', 'tubingen', 'hamburg', 'darmstadt', 'erfurt', 'aachen']
                       },
            'split2': {'test': ['munster', 'lindau', 'frankfurt'],
                       'validation': ['strasbourg', 'weimar', 'aachen'],
                       'train_unsup': ['hamburg', 'cologne', 'monchengladbach', 'tubingen', 'jena', 'darmstadt',
                                       'erfurt', 'dusseldorf'],
                       'train_disc': ['krefeld', 'zurich', 'bremen', 'ulm', 'hanover', 'stuttgart', 'bochum'],
                       'train_sup': ['cologne', 'bochum', 'erfurt', 'jena', 'monchengladbach', 'hamburg', 'tubingen']
                       },
        },

        # ------------------------------------------------------------------------------------------------------------
        # splits with just 8 cities in the labeled dataset

        '8vols': {
            'split0': {'test': ['munster', 'lindau', 'frankfurt'],
                       'validation': ['krefeld', 'zurich', 'bremen'],
                       'train_unsup': ['strasbourg', 'weimar', 'aachen', 'tubingen', 'jena', 'bochum',
                                       'darmstadt', 'dusseldorf'],
                       'train_disc': ['hamburg', 'cologne', 'monchengladbach', 'ulm', 'hanover', 'stuttgart', 'erfurt'],
                       'train_sup': ['strasbourg', 'weimar', 'aachen', 'tubingen', 'jena', 'bochum', 'darmstadt', 'dusseldorf'],
                       },
            'split1': {'test': ['munster', 'lindau', 'frankfurt'],
                       'validation': ['stuttgart', 'zurich', 'weimar'],
                       'train_unsup': ['cologne', 'hamburg', 'aachen', 'monchengladbach', 'ulm', 'bochum',
                                       'darmstadt', 'erfurt'],
                       'train_disc': ['krefeld', 'bremen', 'strasbourg', 'tubingen', 'jena', 'hanover', 'dusseldorf'],
                       'train_sup': ['cologne', 'hamburg', 'aachen', 'monchengladbach', 'ulm', 'bochum',
                                     'darmstadt', 'erfurt'],
                       },
            'split2': {'test': ['munster', 'lindau', 'frankfurt'],
                       'validation': ['strasbourg', 'weimar', 'aachen'],
                       'train_unsup': ['hamburg', 'cologne', 'monchengladbach', 'tubingen', 'jena', 'darmstadt',
                                       'erfurt', 'dusseldorf'],
                       'train_disc': ['krefeld', 'zurich', 'bremen', 'ulm', 'hanover', 'stuttgart', 'bochum'],
                       'train_sup': ['hamburg', 'cologne', 'monchengladbach', 'tubingen', 'jena', 'darmstadt',
                                     'erfurt', 'dusseldorf']
                       },
        },

        # -----------------
        # All the data:

        'all_data': {'all': ['strasbourg', 'weimar', 'aachen', 'tubingen', 'jena', 'bochum', 'darmstadt',
                             'dusseldorf', 'hamburg', 'cologne', 'monchengladbach', 'ulm', 'hanover',
                             'stuttgart', 'erfurt', 'krefeld', 'zurich', 'bremen', 'munster', 'lindau',
                             'frankfurt', 'krefeld', 'zurich', 'bremen']}
    }
    return splits


if __name__ == '__main__':
    _splits = get_splits()
    for k, v in zip(_splits.keys(), _splits.values()):
        print('\n' + '- ' * 20)
        print('number of volumes: {0}'.format(k))
        print('values:', v)
