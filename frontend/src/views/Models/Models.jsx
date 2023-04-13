import { IconButton, Button, Flex, HStack, Stack, Input } from "@chakra-ui/react";
import { useCompositeState } from "ds4biz-core";
import { useContext } from "react";
import { CLIENT, StateContext } from "../../config/constants";
import { Model } from "./Model";
import { ModelCreation } from "./ModelCreation";
import { Report } from "./Report";
import { RiAddFill, RiDownloadCloud2Line, RiFileChartLine } from 'react-icons/ri';
import React, { useState, useRef } from 'react';
import axios from 'axios';
import { saveAs } from 'file-saver';


function str2bytes (str) {
    var bytes = new Uint8Array(str.length);
    for (var i=0; i<str.length; i++) {
        bytes[i] = str.charCodeAt(i);
    }
    return bytes;
}

export function Models({ models }) {

  const state = useCompositeState({ view: "list", fdata: null });
  const _state = useContext(StateContext);
  const ref_import = useRef();
  const ref_report = useRef();
  console.log('models');
  switch (state.view) {
    case "list":
      return (
        <Stack w="100%" h="100%" spacing="1rem">
          <HStack>
            <Button fontSize={"15px"} leftIcon={<RiAddFill />}
                border='0px' variant='outline' color='#A9A9A9'
                onClick={(e) =>(state.view = "new")}>
             New
             </Button>
             <Button fontSize={"15px"} leftIcon={<RiDownloadCloud2Line />}
                onClick={(e) => {
                    console.log('click');
                    ref_import.current.click()
                }}
                border='0px' variant='outline' color='#A9A9A9'>
             Import
             </Button>
             <input
                type='file'
                accept=".zip"
                ref={ref_import}
                onChange={(e) => {
                    console.log('change import');
                    console.log(event.target.files[0]);
                    const formData = new FormData();
                    formData.append('file', event.target.files[0]);
                    CLIENT.import.post(formData).then(()=>location.reload()).catch((err) => console.log(err));
                }}
                onSubmit={(e) => {
                    e.preventDefault();
                    console.log('submit import');
                    _state.refresh = new Date();
                    // location.reload();
                }}
                style={{ display: 'none' }}/>


            <Button fontSize={"15px"} leftIcon={<RiFileChartLine />}
                onClick={(e) =>(
                ref_report.current.click()
                )}
                border='0px' variant='outline' color='#A9A9A9'>
             Report
            </Button>
            <input
                type='file'
                accept=".eval"
                ref={ref_report}
                onChange={(e)=>{
                    const fileReader = new FileReader();
                    const { files } = event.target;
                    fileReader.readAsText(files[0], "UTF-8");
                    fileReader.onload = e => {
                      const content = e.target.result;
                      state.fdata = JSON.parse(content);
                    };
                    state.view = "report";
                    console.log('change report');
                }}
                onSubmit={(e)=>{
                    e.preventDefault();
                    console.log('submit report');
                }}
                style={{ display: 'none' }}/>
          </HStack>
          <Stack>
            {models.map((name) => (
              <Model
                //onClick={(e) => (state.view = "model")}
                name={name}
                key={name}
                _state={_state}
                onDelete={(e) =>
                  CLIENT[name].delete().then((resp) => {
                    _state.refresh = new Date();
                  })
                }
                onExport={(e) =>
                  CLIENT[name].export.get({responseType: "arraybuffer"})
                  .then(response => {
                    console.log('download');
                    console.log(response);
                    const blob = new Blob([response.data], {
                            type: 'application/octet-stream'
                            })
                    return blob
                    })
                    .then(blob => {
                    console.log(blob)
                    const filename = name+'.zip'
                    saveAs(blob, filename)
                    console.log('hello');
                    })
                  .catch(error => {
                    console.log(error);
                    })
                   }
              />
            ))}
          </Stack>
        </Stack>
      );
    case "new":
      return (
          <ModelCreation onClose={(e) => (state.view = "list")} />
      );
    case "report":
      return (
          <Report data={state.fdata} onClose={(e) => (state.view = "list")} />
      );


    default:
      break;
  }
}
