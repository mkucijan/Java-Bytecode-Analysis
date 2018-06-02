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

package org.apache.commons.beanutils;

import java.util.Hashtable;


/**
 * Used to test
 *
 * @version $Id: ExtendMapBean.java 1632171 2014-10-15 20:15:17Z ggregory $
 */

public class ExtendMapBean extends Hashtable<Object, Object> {

    private String dbName = "[UNSET]";

    public ExtendMapBean() {}

    public String getUnusuallyNamedProperty()
    {
        return dbName;
    }

    public void setUnusuallyNamedProperty(final String dbName)
    {
        this.dbName = dbName;
    }

}
