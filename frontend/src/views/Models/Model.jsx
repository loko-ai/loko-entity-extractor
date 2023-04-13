import { Box, HStack, IconButton, Spacer, Stack, Tag, Modal, ModalOverlay, ModalContent, ModalHeader,
ModalCloseButton, ModalBody, ModalFooter, Button, Input } from "@chakra-ui/react";
import { RiDeleteBin4Line, RiShareForward2Fill, RiFileCopy2Fill } from "react-icons/ri";
import { CLIENT, StateContext, baseURL } from "../../config/constants";
import { useEffect, useState } from "react";
import urlJoin from 'url-join';
import { useCompositeState } from "ds4biz-core";

function CopyModel({ state, _state }) {

    var new_name = '';

    return (
    <Modal blockScrollOnMount={false} isOpen={state.open_copy_model} onClose={()=>state.open_copy_model=false}>
        <ModalOverlay />
        <ModalContent color='#A9A9A9' bg='#222222'>
          <ModalHeader>Copy {state.model.identifier}</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Input placeholder='new name' size='md' onChange={(e)=>{
                new_name=e.target.value;
                console.log(new_name);
            }}/>
          </ModalBody>

          <ModalFooter>
            <Button variant='ghost' bg="#333333" onClick={(e) =>
                    CLIENT[state.model.identifier].copy.get({ params: { new_name: new_name } })
                  .then((resp) => {
                    state.open_copy_model=false;
                    _state.refresh = new Date();
                  })}>
            Save
            </Button>&nbsp;
            <Button bg="#333333" mr={3} onClick={()=>state.open_copy_model=false}>
              Close
            </Button>

          </ModalFooter>
        </ModalContent>
      </Modal>
    );
}


export function Model({ name, _state, onDelete, onExport,...rest }) {
    const state = useCompositeState({
    model: null,
    open_copy_model: false
  });

    useEffect(() => {
      CLIENT[name]
      .get()
      .then((resp) => (state.model=resp.data))
      .catch((err) => console.log(err));
    }, []);

  if (!state.model) return null;

  var color = "orange.400";
  var tag = "";
  var model_name = "";

  if (!state.model.is_trainable) {
    tag = "Pretrained";
    color = "green.200";
  } else if (state.model.is_trained) {
    tag = "Fitted";
    color = "orange.200";
  } else {
    tag = "Not fitted"
    color = "red.500"
  }

  if (state.model.type.includes("Spacy")) {
    model_name = "Spacy";
  } else if (state.model.type.includes("HF")) {
    model_name = "HF";
  } else {
    model_name = "Rules";
  }

  return (
   <a href={urlJoin(baseURL, name)} target="_blank">
    <HStack
      bg="#222222"
      color='white'
      borderRadius={"10px"}
      w="100%"
      py="0.5rem"
      px="1rem"
      {...rest}
    >
      <Stack spacing={5}>
        <HStack color={"red.600"}>
          <Box><b>{name}</b></Box>
          <Tag borderRadius={"10px"} p=".1rem" bg={color} fontSize="10">
            <b>{tag}</b>
          </Tag>
        </HStack>
        <HStack fontSize={"xs"} alignItems="start" spacing="5">
          <Stack spacing="0" w="40px">
            <Box><b>Type</b></Box>
            <Box color="#A9A9A9">{model_name}</Box>
          </Stack>
          <Stack spacing="0">
            <Box><b>Lang</b></Box>
            <Box color="#A9A9A9">{state.model.lang}</Box>
          </Stack>
          <Stack spacing="0">
            <Box><b>Creation</b></Box>
            <Box color="#A9A9A9">{state.model.date_of_creation}</Box>
          </Stack>
          <Stack spacing="0" w="300px">
            <Box><b>Tags</b></Box>
            <Box color="#A9A9A9">{state.model.tags.toString().replace(/,/g, ", ")}</Box>
          </Stack>
        </HStack>
      </Stack>
      <Spacer />
      {(() => {
                if (state.model.is_trainable) {
                    return (<HStack>
      <IconButton
          size="sm"
          borderRadius={"full"}
          bg='#333333'
          icon={<RiFileCopy2Fill />}
          onClick={(e) => {
            e.stopPropagation();
            e.preventDefault();
            state.open_copy_model = true;
          }}
        />
        <CopyModel state={state} _state={_state} />
        <IconButton
          size="sm"
          borderRadius={"full"}
          bg='#333333'
          icon={<RiShareForward2Fill />}
          onClick={(e) => {
            e.stopPropagation();
            e.preventDefault();
            onExport(e);
          }}
        />
        <IconButton
          size="sm"
          borderRadius={"full"}
          bg='#333333'
          icon={<RiDeleteBin4Line />}
          onClick={(e) => {
            e.stopPropagation();
            e.preventDefault();
            onDelete(e);
          }}
        />
      </HStack>)
                }
            })()}
    </HStack>
    </a>
  );
}
