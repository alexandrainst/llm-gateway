# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""
Add a new task (required arguments):

TASK_NAME: {
    'tokens_to_generate': how many tokens we want to generate.
    'template': the template with at least {context} and {query}.
}
"""

TASKS = {
    'niah': {
        'tokens_to_generate': 128,
        'template': """Nogle særlige magiske {type_needle_v} er skjult i teksten nedenfor. Husk dem. Senere vil jeg spørge til {type_needle_v}.\n{context}\nHvilke særlige magiske {type_needle_v} for {query} nævnes i den givne tekst?""",
        'answer_prefix': """ De særlige magiske {type_needle_v} for {query} i den givne tekst er"""
    },
    
    'variable_tracking': {
        'tokens_to_generate': 30,
        'template': """Husk og følg kæderne af variabeltildelinger, som er skjult i teksten nedenfor.\n\n{context}\nSpørgsmål: Find alle variabler, der får værdien {query} i teksten ovenfor.""",
        'answer_prefix': """ Svar: Ifølge kæderne af variabeltildelinger i teksten ovenfor får {num_v} variabler værdien {query}; de er: """
    },
    
    'common_words_extraction': {
        'tokens_to_generate': 120,
        'template': """Nedenfor er en nummereret liste af ord. Nogle ord forekommer oftere end andre. Husk de ord, der forekommer oftest.\n{context}\nSpørgsmål: Hvilke {num_cw} ord er mest almindelige i listen ovenfor?""",
        'answer_prefix': """ Svar: De {num_cw} mest almindelige ord i listen er:"""
    },
    
    'freq_words_extraction' : {
        'tokens_to_generate': 50,
        'template': """Læs den følgende kodede tekst og hold styr på hyppigheden af hvert kodet ord. Find de tre hyppigst forekommende kodede ord. {context}\nSpørgsmål: Giv ingen forklaring. Ignorér venligst punktummerne '....'. Hvilke tre ord forekommer hyppigst i den kodede tekst ovenfor?""",
        'answer_prefix': """ Svar: Ifølge den kodede tekst ovenfor er de tre hyppigst forekommende ord:"""
    },

    'qa': {
        'tokens_to_generate': 32, 
        'template': """Besvar spørgsmålet ud fra de givne dokumenter. Giv kun svaret og ingen andre ord.\n\nFølgende dokumenter er givet.\n\n{context}\n\nBesvar spørgsmålet ud fra de givne dokumenter. Giv kun svaret og ingen andre ord.\n\nSpørgsmål: {query}""",
        'answer_prefix': """ Svar:""",
    },
}
