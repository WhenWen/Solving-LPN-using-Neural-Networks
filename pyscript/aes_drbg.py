#MIT License

#Copyright (c) 2019 Samuele Cornell

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.


import pyaes


class AES_DRBG(object):
    """AES based DRBG class compliant with SP-900 80A NIST standard.
       This implementation follows closely the above Specification in order to be as clear as possible.

       Constructor requires only the specification of the desired AES version (128/192 or 256 Keylen):

       Parameters
       ----------
       keylen : keylength in bits for AES block cipher used in the DRBG

       Returns
       -------
       drbg_object


       The DRBG has 3 main methods:

             1 INSTANTIATE: initialize and instantiate the DRBG
             2 RESEED: reseed the DRBG (this must be done every 2**48 calls)
             3 GENERATE obtain pseudo-random bits from DRBG

    """

    def __init__(self, keylen):

        self.keylen = keylen

        self.reseed_counter = 0
        self.key = False
        self.V = False

        self.outlen = 16  # same for all

        if keylen == 256:
            self.seedlen = 48
            self.keylen = 32


        elif keylen == 192:
            self.seedlen = 40
            self.keylen = 24

        elif keylen == 128:
            self.seedlen = 32
            self.keylen = 16


        else:
            raise ValueError("Keylen not supported")

        self.reseed_interval = 2 ** 48  # same for all

    def instantiate(self, entropy_in, per_string=b''):
        '''
            Method handling initialization of the DRBG (see Specification)

           Parameters
           ----------
           entropy_in : hex byterray (e.g. \xFF\xF1....etc (len = seedlen_bits /8))
                       full entropy seed for DRBG, it must be seedlen bits

           per_string : hex byterray (e.g. \xFF\xF1....etc (len must be less or equal seedlen))
                       additional input which will be xored with the input entropy for added security (optional)

           Returns
           -------


        '''

        if len(per_string) != 0:

            temp = len(per_string)  # NB len is in bytes

            if temp < self.seedlen:

                per_string = per_string + b"\x00" * (self.seedlen - temp)  # pad

            else:

                raise ValueError("Length of personalization string must be equal or less than seedlen")

        else:

            per_string = b"\x00" * self.seedlen

        seed_material = int(entropy_in.hex(), 16) ^ int(per_string.hex(), 16)
        seed_material = seed_material.to_bytes(self.seedlen, byteorder='big', signed=False)

        self.key = b"\x00" * self.keylen
        self.V = b"\x00" * self.outlen

        self.aes = pyaes.AESModeOfOperationECB(self.key)

        self._update(seed_material)

        self.reseed_counter = 1

    def _update(self, provided_data):
        '''
            DRBG internal Update function (see Specification)

            Parameters
            ----------
            provided_data : hex byterray (e.g. \xFF\xF1....etc (len = seedlen_bits /8))
                            input data to the update function (it is ensured in other methods is seedlen bits)


            Returns
            -------

        '''

        temp = b""

        while (len(temp) < self.seedlen):
            # increment V
            self.V = (int(self.V.hex(), 16) + 1) % 2 ** (self.outlen * 8)
            self.V = self.V.to_bytes(self.outlen, byteorder='big', signed=False)

            output_block = self.aes.encrypt(self.V)  # generate keystream

            temp = temp + output_block  # concat keystream

        temp = temp[0:self.seedlen]

        temp = int(temp.hex(), 16) ^ int(provided_data.hex(), 16)  # xor keystream
        temp = temp.to_bytes(self.seedlen, byteorder='big', signed=False)

        self.key = temp[0:self.keylen]

        self.V = temp[-self.outlen:]

        self.aes = pyaes.AESModeOfOperationECB(self.key)  # update the key

    def reseed(self, entropy_in, add_in=b''):

        '''
            DRBG Reseed function (see Specification)
            Similar to instantiate except the previous DRBG state (self.key, self.V, self.reseed_counter) is
            preserved and updated with full entropy.

            Parameters
            ----------
            entropy_in : hex byterray (e.g. \xFF\xF1....etc (len = seedlen_bits /8))
                       full entropy seed for DRBG, it must be seedlen bits

            add_in : hex byterray (e.g. \xFF\xF1....etc (len must be less or equal seedlen))
                    additional input which will be xored with the input entropy for added security (optional)

            Returns
            -------
        '''

        if len(add_in) != 0:

            temp = len(add_in)  # NB len is in bytes

            if temp < self.seedlen:

                add_in = add_in + b"\x00" * (self.seedlen - temp)  # pad

            else:

                raise ValueError("Length of personalization string must be equal or less than seedlen")

        else:

            add_in = b"\x00" * self.seedlen

        seed_material = int(entropy_in.hex(), 16) ^ int(add_in.hex(), 16)
        seed_material = seed_material.to_bytes(self.seedlen, byteorder='big', signed=False)

        self._update(seed_material)

        self.reseed_counter = 1

    def generate(self, req_bytes, add_in=b''):
        ''' DRBG Generate Funtion (see Specification)
            returns req_bytes pseudo-random bits from the DRBG

            Parameters
            ----------
            req_bytes : int
                       number of bytes requested from the DRBG

            add_in : hex byterray (e.g. \xFF\xF1....etc (len must be less or equal seedlen))
                    additional input which will be xored with the output of DRBG (optional)

            Returns
            -------
            returned_bytes: hex byterray (e.g. \xFF\xF1....etc (len is req_bytes))
                    pseudo-random bytes from DRBG ready to be used in whatever application

        '''


        if self.reseed_counter > self.reseed_interval:
            raise Warning("the DBRG should be reseeded !!!")

        if len(add_in) != 0:

            temp = len(add_in)

            if temp < self.seedlen:
                add_in = add_in + b"\x00" * (temp - self.seedlen)

            self._update(add_in)
        else:

            add_in = b"\x00" * self.seedlen

        temp = b''

        while (len(temp) < req_bytes):
            self.V = (int(self.V.hex(), 16) + 1) % 2 ** (self.outlen * 8)
            self.V = self.V.to_bytes(self.outlen, byteorder='big', signed=False)

            output_block = self.aes.encrypt(self.V)

            temp = temp + output_block

        returned_bytes = temp[0:req_bytes]

        self._update(add_in)

        self.reseed_counter = self.reseed_counter + 1

        return returned_bytes
