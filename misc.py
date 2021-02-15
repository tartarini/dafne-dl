#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2021 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import hashlib
import os


def calculate_file_hash(file_path, cache_results=False, force_rewrite_cache=False):
    if not cache_results:
        return hashlib.sha256(open(file_path, 'rb').read()).hexdigest()
    
    # check if the hash exists on disk
    hash_file = file_path + '.sha256'
    if not force_rewrite_cache:
        try:
            with open(hash_file, 'r') as f:
                output_hash = f.readline().strip()
        except OSError:
            print('Error while reading from hash file')
        else:
            if len(output_hash) == 64:
                print('Using cached hash')
                return output_hash
            else:
                print('Invalid stored hash')

    # file does not exist, or stored hash is invalid
    output_hash = calculate_file_hash(file_path, False)  # fallback to calculating hash
    try:
        with open(hash_file, 'w') as f:
            f.write(output_hash)
    except OSError:
        print('Error writing hash to file')
    return output_hash

