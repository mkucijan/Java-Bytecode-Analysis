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
package org.apache.commons.rng.core.source32;

import java.util.Random;

import org.junit.Assert;
import org.junit.Test;

public class JDKRandomTest {
    @Test
    public void testReferenceCode() {
        final long refSeed = -1357111213L;
        final JDKRandom rng = new JDKRandom(refSeed);
        final Random jdk = new Random(refSeed);

        // This is a trivial test since "JDKRandom" delegates to "Random".

        final int numRepeats = 1000;
        for (int r = 0; r < numRepeats; r++) {
            Assert.assertEquals(r + " nextInt", jdk.nextInt(), rng.nextInt());
        }
    }
}
