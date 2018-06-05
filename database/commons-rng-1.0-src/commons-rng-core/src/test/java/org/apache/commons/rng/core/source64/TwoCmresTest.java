/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.commons.rng.core.source64;

import org.junit.Assert;
import org.junit.Test;


public class TwoCmresTest {
    @Test
    public void testAsymmetric() {
        final int index1 = 2;
        final int index2 = 5;
        final int seed = -123456789;

        final TwoCmres rng1 = new TwoCmres(seed, index1, index2);
        final TwoCmres rng2 = new TwoCmres(seed, index2, index1);

        // Try a few values.
        final int n = 1000;
        for (int i = 0; i < n; i++) {
            Assert.assertNotEquals("i=" + i, rng1.nextLong(), rng2.nextLong());
        }
    }

    @Test
    public void testSubcycleGeneratorsMustBeDifferent() {
        final int max = TwoCmres.numberOfSubcycleGenerators();
        for (int i = 0; i < max; i++) {
            try {
                new TwoCmres(-97845, i, i);
                Assert.fail("Exception expected");
            } catch (IllegalArgumentException e) {
                // Expected.
            }
        }
    }

    @Test
    public void testSubcycleGeneratorsIndex() {
        final int seed = 246810;

        // Valid indices are between 0 (included) and max (excluded).
        final int max = TwoCmres.numberOfSubcycleGenerators();

        for (int i = 0; i < max; i++) {
            for (int j = 0; j < max; j++) {
                if (i != j) { // Subcycle generators must be different.
                    // Can be instantiated.
                    new TwoCmres(seed, i, j);
                }
            }
        }

        for (int wrongIndex : new int[] { -1, max }) {
            try {
                new TwoCmres(seed, wrongIndex, 1);
                Assert.fail("Exception expected for index=" + wrongIndex);
            } catch (IndexOutOfBoundsException e) {
                // Expected.
            }

            try {
                new TwoCmres(seed, 1, wrongIndex);
                Assert.fail("Exception expected for index=" + wrongIndex);
            } catch (IndexOutOfBoundsException e) {
                // Expected.
            }
        }
    }
}
