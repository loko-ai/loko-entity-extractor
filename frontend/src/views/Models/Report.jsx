import {
  Flex,
  Button,
  IconButton,
  Stack,
  HStack,
  VStack,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  Text,
  Thead,
  Tbody,
  Table,
  Tr,
  Th,
  Td,
  Box
} from "@chakra-ui/react";
import { useContext } from "react";
import { CLIENT, StateContext } from "../../config/constants";
import { BaseForm } from "./BaseForm";
import { RiArrowLeftLine } from "react-icons/ri";
import { Heatmap, Bar } from '../../plots';


export function Report({ data, onClose, ...rest }) {
  const _state = useContext(StateContext);
  const chartMargins = { top: 20, right: 10, bottom: 70, left: 20 };
  if (!data) return null;
  let info = data.info;
  const metrics = data.test_report.metrics;
  const cm = data.test_report.confusion_matrix.values;
  const labels = data.test_report.confusion_matrix.labels;
  const creport = data.test_report.classification_report;
  const creport_cols = Object.keys(creport[0]);
  const distro = data.distro;
  delete distro.O;
  const info_t = (({identifier, is_trainable, is_trained, 
                    tags, output_dir,_args, 
                    current_epoch, ...o}) => o)(info) 
  let model_name = info.identifier;

  console.log(info);

  return (
    <Stack w="100%" h="100%" spacing="1rem" color='#A9A9A9'>
    <IconButton
          align="left"
          size="sm"
          w="30px"
          h="30px"
          borderRadius={"full"}
          bg='#222222'
          icon={<RiArrowLeftLine />}
          onClick={onClose}
        />
        <HStack w="100%" h="50vh" spacing="1rem">
            <Flex bg="#222222" color='white' borderRadius={"10px"} w="100%" h="100%">
                <VStack w="100%" h="100%" spacing=".1rem">
                    <Text fontSize="md" color='#A9A9A9' pl="1rem" pt="1rem" pb="1rem" w="100%"><b> Model information </b></Text>
                    <Text fontSize="12px" color='#A9A9A9' pl="1.5rem" pt=".2rem" w="100%" key="1"><b>name</b>: {model_name}</Text>
                    {Object.entries(info_t).map((k,i) => {
                                    return <Text fontSize="12px" color='#A9A9A9' pl="1.5rem" pt=".2rem" w="100%" key={1+i}><b>{k[0]}</b>: {k[1].toString()}</Text>
                                    })}
                </VStack>
            </Flex>
            <Flex bg="#222222" color='white' borderRadius={"10px"} w="100%" h="100%">
                <VStack w="100%" h="100%" spacing=".1rem">
                    <Text fontSize="md" color='#A9A9A9' pl="1rem" pt="1rem" w="100%"><b> Distribution </b></Text>
                    <Bar data={distro} />
                </VStack>
            </Flex>
        </HStack>
        <HStack w="100%" h="60vh" spacing="1rem">
            <Flex bg="#222222" color='white' borderRadius={"10px"} w="100%" h="100%">
                <VStack w="100%" h="100%" spacing=".1rem">
                    <Text fontSize="md" color='#A9A9A9' pl="1rem" pt="1rem" w="100%"><b> Confusion Matrix </b></Text>
                    <Heatmap data={cm} labels={labels} />
                </VStack>
            </Flex>

            <Flex bg="#222222" color='white' borderRadius={"10px"} w="100%" h="100%">
                <VStack w="100%" h="100%" spacing="2rem">
                    <Text fontSize="md" color='#A9A9A9' pl="1rem" pt="1rem" w="100%"><b> Classification report </b></Text>
                    <Text fontSize="12px" color='#A9A9A9' pl="2rem" pt=".2rem" w="100%" key="1"><b>accuracy</b>: {metrics.accuracy}</Text>
                    <Table size="sm" color='#A9A9A9' w="90%" h="70%">
                        <Thead>
                            <Tr>
                                {creport_cols.map((el, i) => (
                                    <Td borderColor="gray.600" fontSize={"xs"} key={i}>
                                    <b>
                                    {el}
                                    </b>
                                    </Td>))}
                            </Tr>
                        </Thead>
                        <Tbody>
                            {creport.map((row, i) => {
                                return(
                                    <Tr key={i+1}>
                                    {Object.values(row).map((c, j) => (<Td borderColor="gray.600" fontSize={"xs"} key={i+j+5}>
                                                            {c}
                                                       </Td>))}
                                    </Tr>)})}
                        </Tbody>
                    </Table>
                </VStack>
            </Flex>
        </HStack>
    </Stack>
  );
}