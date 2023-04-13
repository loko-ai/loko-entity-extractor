import React from "react";
import { URLObject } from "../hooks/urls";

const base_transformer = {};

const base_model = {"typology": "trainable_spacy",
        "lang": "it",
        "extend_pretrained": true,
        "n_iter": 100,
        "minibatch_size": 500,
        "dropout_rate": 0.2};
const baseURL = import.meta.env.VITE_BASE_URL || "/";

const CLIENT = new URLObject(baseURL);
const StateContext = React.createContext();

export { StateContext, CLIENT, baseURL, base_transformer, base_model };
